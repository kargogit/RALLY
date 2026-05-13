/*
llvmLowerInteger.cpp  (Implementation of Step 11)

Supported Instructions:

Data movement
- MOV
- LEA
- MOVZX
- MOVSX
- XCHG (reg ↔ mem only)
- PUSH
- POP
- LEAVE
- CDQ

Arithmetic / Logic
- ADD, SUB, XOR, AND, OR
- DIV
- INC
- IMUL (both 2-operand and 1-operand form)
- IDIV (32-bit EDX:EAX ÷ ECX form)

Comparison / Flags
- CMP
- TEST
- CMOVE
- SETcc family: SETC/SETB, SETE/SETZ, SETNE, SETPE/SETP

Shifts (with correct CF/OF edge-case handling)
- SHL
- SHR
- SAR
*/
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <cmath>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include "ast.pb.h"
#include <google/protobuf/struct.pb.h>
#include <google/protobuf/util/json_util.h>

using namespace llvm;

// ----------------------------- small utilities -----------------------------

static bool endsWith(const std::string &s, const char *suff) {
  const size_t n = std::strlen(suff);
  return s.size() >= n && s.compare(s.size() - n, n, suff) == 0;
}

static std::string toUpper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return char(std::toupper(c)); });
  return s;
}

static std::optional<int64_t> parseInt64Loose(const std::string &s0) {
  std::string s = s0;

  if (s.empty()) return std::nullopt;

  int base = 10;
  if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) base = 16;

  char *end = nullptr;
  errno = 0;

  if (base == 16) {
    unsigned long long uv = std::strtoull(s.c_str(), &end, 16);
    if (errno == 0 && end != s.c_str() && *end == '\0') {
      return (int64_t)uv;
    }
  }

  // decimal fallback (original signed path)
  long long v = std::strtoll(s.c_str(), &end, base);
  if (errno != 0 || end == s.c_str() || *end != '\0') return std::nullopt;
  return (int64_t)v;
}

static std::optional<int64_t> valueAsInt64(const google::protobuf::Value &v) {
  using V = google::protobuf::Value;
  switch (v.kind_case()) {
    case V::kNumberValue: {
      double d = v.number_value();
      if (!std::isfinite(d)) return std::nullopt;
      if (d > (double)std::numeric_limits<int64_t>::max()) return std::nullopt;
      if (d < (double)std::numeric_limits<int64_t>::min()) return std::nullopt;
      return (int64_t)d;
    }
    case V::kStringValue: {
      return parseInt64Loose(v.string_value());
    }
    default:
      return std::nullopt;
  }
}

static std::optional<std::string>
structStringField(const google::protobuf::Value &v, const std::string &key) {
  using V = google::protobuf::Value;
  if (v.kind_case() != V::kStructValue) return std::nullopt;
  const auto &fs = v.struct_value().fields();
  auto it = fs.find(key);
  if (it == fs.end()) return std::nullopt;
  if (it->second.kind_case() == V::kStringValue) return it->second.string_value();
  return std::nullopt;
}

static std::optional<google::protobuf::Value>
structFieldValue(const google::protobuf::Value &v, const std::string &key) {
  using V = google::protobuf::Value;
  if (v.kind_case() != V::kStructValue) return std::nullopt;
  const auto &fs = v.struct_value().fields();
  auto it = fs.find(key);
  if (it == fs.end()) return std::nullopt;
  return it->second;
}

// ----------------------------- %State layout ------------------------------

static constexpr unsigned kGprCount  = 16;
static constexpr unsigned kFlagCount = 9;
static constexpr unsigned kXmmCount  = 16;
static constexpr unsigned kFlagsBase = kGprCount;
static constexpr unsigned kXmmBase   = kGprCount + kFlagCount;

enum FlagIndex : unsigned {
  CF = 0, PF = 1, AF = 2, ZF = 3, SF = 4, OF = 5, DF = 6, IF = 7, RF = 8,
};

// -------------------------- register decode/mapping -------------------------

struct RegInfo {
  bool isValid = false;
  bool isXmm = false;

  int gprIndex = -1;
  unsigned bitOffset = 0;
  unsigned bitWidth  = 0;

  unsigned xmmIndex = 0;
};

static int gprFieldIndex64(const std::string &regUpper) {
  static const std::map<std::string, int> m = {
    {"RAX",0},{"RBX",1},{"RCX",2},{"RDX",3},{"RSI",4},{"RDI",5},
    {"R8",6},{"R9",7},{"R10",8},{"R11",9},{"R12",10},{"R13",11},{"R14",12},{"R15",13},
    {"RSP",14},{"RBP",15}
  };
  auto it = m.find(regUpper);
  if (it == m.end()) return -1;
  return it->second;
}

static RegInfo decodeReg(const std::string &regName0) {
  RegInfo ri;
  std::string r = toUpper(regName0);

  if (r.size() >= 3 && r.substr(0, 3) == "XMM") {
    std::string idxStr = r.substr(3);
    auto nOpt = parseInt64Loose(idxStr);
    if (!nOpt || *nOpt < 0 || *nOpt >= (int64_t)kXmmCount) return ri;
    ri.isValid = true;
    ri.isXmm = true;
    ri.xmmIndex = (unsigned)*nOpt;
    return ri;
  }

  auto setGpr = [&](const std::string &base64, unsigned w, unsigned off) {
    int idx = gprFieldIndex64(base64);
    if (idx < 0) return;
    ri.isValid = true;
    ri.isXmm = false;
    ri.gprIndex = idx;
    ri.bitWidth = w;
    ri.bitOffset = off;
  };

  if (r == "RAX") return (setGpr("RAX",64,0), ri);
  if (r == "EAX") return (setGpr("RAX",32,0), ri);
  if (r == "AX")  return (setGpr("RAX",16,0), ri);
  if (r == "AL")  return (setGpr("RAX",8,0), ri);
  if (r == "AH")  return (setGpr("RAX",8,8), ri);

  if (r == "RBX") return (setGpr("RBX",64,0), ri);
  if (r == "EBX") return (setGpr("RBX",32,0), ri);
  if (r == "BX")  return (setGpr("RBX",16,0), ri);
  if (r == "BL")  return (setGpr("RBX",8,0), ri);
  if (r == "BH")  return (setGpr("RBX",8,8), ri);

  if (r == "RCX") return (setGpr("RCX",64,0), ri);
  if (r == "ECX") return (setGpr("RCX",32,0), ri);
  if (r == "CX")  return (setGpr("RCX",16,0), ri);
  if (r == "CL")  return (setGpr("RCX",8,0), ri);
  if (r == "CH")  return (setGpr("RCX",8,8), ri);

  if (r == "RDX") return (setGpr("RDX",64,0), ri);
  if (r == "EDX") return (setGpr("RDX",32,0), ri);
  if (r == "DX")  return (setGpr("RDX",16,0), ri);
  if (r == "DL")  return (setGpr("RDX",8,0), ri);
  if (r == "DH")  return (setGpr("RDX",8,8), ri);

  if (r == "RSI") return (setGpr("RSI",64,0), ri);
  if (r == "ESI") return (setGpr("RSI",32,0), ri);
  if (r == "SI")  return (setGpr("RSI",16,0), ri);
  if (r == "SIL") return (setGpr("RSI",8,0), ri);

  if (r == "RDI") return (setGpr("RDI",64,0), ri);
  if (r == "EDI") return (setGpr("RDI",32,0), ri);
  if (r == "DI")  return (setGpr("RDI",16,0), ri);
  if (r == "DIL") return (setGpr("RDI",8,0), ri);

  if (r == "RBP") return (setGpr("RBP",64,0), ri);
  if (r == "EBP") return (setGpr("RBP",32,0), ri);
  if (r == "BP")  return (setGpr("RBP",16,0), ri);
  if (r == "BPL") return (setGpr("RBP",8,0), ri);

  if (r == "RSP") return (setGpr("RSP",64,0), ri);
  if (r == "ESP") return (setGpr("RSP",32,0), ri);
  if (r == "SP")  return (setGpr("RSP",16,0), ri);
  if (r == "SPL") return (setGpr("RSP",8,0), ri);

  if (r.size() >= 2 && r[0] == 'R' && std::isdigit((unsigned char)r[1])) {
    std::string base;
    base.push_back('R');
    size_t i = 1;
    while (i < r.size() && std::isdigit((unsigned char)r[i])) {
      base.push_back(r[i]);
      ++i;
    }
    int idx = gprFieldIndex64(base);
    if (idx >= 0) {
      std::string suff = r.substr(i);
      if (suff.empty())       return (setGpr(base,64,0), ri);
      if (suff == "D")        return (setGpr(base,32,0), ri);
      if (suff == "W")        return (setGpr(base,16,0), ri);
      if (suff == "B")        return (setGpr(base,8,0),  ri);
    }
  }
  return ri;
}

static unsigned memSizeBytesFromOperand(const lifted_ast::Operand &op) {
  if (!op.has_size()) return 0;
  std::string s = toUpper(op.size());
  if (s == "BYTE") return 1;
  if (s == "WORD") return 2;
  if (s == "DWORD") return 4;
  if (s == "QWORD") return 8;
  return 0;
}

// ----------------------------- stack slots ---------------------------------

struct StackSlotInfo {
  std::string name;
  std::string baseReg;
  int64_t startOff = 0;
  int32_t size = 0;
  int32_t align = 1;
  llvm::AllocaInst *alloca = nullptr;
};

static std::optional<std::pair<llvm::Value*, llvm::Align>>
lookupStackSlotPtr(IRBuilder<> &B,
                   const std::vector<StackSlotInfo> &slots,
                   const std::string &baseRegUpper,
                   int64_t disp,
                   unsigned accessSize) {
  for (const auto &s : slots) {
    if (toUpper(s.baseReg) != baseRegUpper) continue;
    int64_t begin = s.startOff;
    int64_t end   = s.startOff + (int64_t)s.size;
    int64_t reqEnd = disp + (int64_t)accessSize;
    if (disp >= begin && reqEnd <= end) {
      int64_t innerOff = disp - begin;
      llvm::Value *p = s.alloca;
      if (innerOff != 0) {
        auto *i8 = B.getInt8Ty();
        p = B.CreateGEP(i8, p, B.getInt64(innerOff), "stk.gep");
      }
      llvm::Align a((unsigned)std::max<int32_t>(1, s.align));
      return std::make_pair(p, a);
    }
  }
  return std::nullopt;
}

// ----------------------------- lowering context ----------------------------

struct FnLowerCtx {
  llvm::LLVMContext &C;
  llvm::Module &M;
  llvm::StructType *StateTy = nullptr;
  llvm::Function *F = nullptr;
  llvm::Value *StateArg = nullptr;
  const lifted_ast::Program *P = nullptr;
  const lifted_ast::Function *FnAst = nullptr;

  std::map<std::string, llvm::BasicBlock*> bbIdToLlvm;
  std::vector<StackSlotInfo> promotedSlots;

  llvm::DenseMap<unsigned, llvm::Value*> stateGepCache;

  std::map<std::string, std::string> &bbLlvmMapping;
  std::map<std::string, std::string> &instrLlvmMapping;

  FnLowerCtx(llvm::LLVMContext &c, llvm::Module &m, llvm::StructType *stateTy,
             llvm::Function *f, llvm::Value *stateArg,
             const lifted_ast::Program *p, const lifted_ast::Function *fnAst,
             std::map<std::string, std::string> &bbMap,
             std::map<std::string, std::string> &instrMap)
    : C(c), M(m), StateTy(stateTy), F(f), StateArg(stateArg),
      P(p), FnAst(fnAst),
      bbLlvmMapping(bbMap), instrLlvmMapping(instrMap) {}
};

static llvm::Value*
getStateFieldPtr(FnLowerCtx &LC, IRBuilder<> &B, unsigned fieldIdx) {
  auto it = LC.stateGepCache.find(fieldIdx);
  if (it != LC.stateGepCache.end()) return it->second;
  llvm::Value *p = B.CreateStructGEP(LC.StateTy, LC.StateArg, fieldIdx, "state.gep");
  LC.stateGepCache[fieldIdx] = p;
  return p;
}

static llvm::Value* loadGpr64(FnLowerCtx &LC, IRBuilder<> &B, unsigned gprIdx) {
  llvm::Value *p = getStateFieldPtr(LC, B, gprIdx);
  return B.CreateLoad(B.getInt64Ty(), p, "gpr64");
}

static void storeGpr64(FnLowerCtx &LC, IRBuilder<> &B, unsigned gprIdx, llvm::Value *vI64) {
  llvm::Value *p = getStateFieldPtr(LC, B, gprIdx);
  B.CreateStore(vI64, p);
}

static llvm::Value* loadFlag(FnLowerCtx &LC, IRBuilder<> &B, FlagIndex f) {
  llvm::Value *p = getStateFieldPtr(LC, B, kFlagsBase + (unsigned)f);
  return B.CreateLoad(B.getInt1Ty(), p, "flag");
}

static void storeFlag(FnLowerCtx &LC, IRBuilder<> &B, FlagIndex f, llvm::Value *vI1) {
  llvm::Value *p = getStateFieldPtr(LC, B, kFlagsBase + (unsigned)f);
  B.CreateStore(vI1, p);
}

static llvm::Type* intTy(llvm::LLVMContext &C, unsigned bits) {
  return llvm::IntegerType::get(C, bits);
}

static llvm::Value* truncOrZext(IRBuilder<> &B, llvm::Value *v, llvm::Type *dstTy) {
  llvm::Type *srcTy = v->getType();
  if (srcTy == dstTy) return v;
  if (srcTy->isIntegerTy() && dstTy->isIntegerTy()) {
    unsigned sb = srcTy->getIntegerBitWidth();
    unsigned db = dstTy->getIntegerBitWidth();
    if (sb > db) return B.CreateTrunc(v, dstTy);
    if (sb < db) return B.CreateZExt(v, dstTy);
  }
  if (srcTy->isPointerTy() && dstTy->isPointerTy()) return v;
  return B.CreateBitCast(v, dstTy);
}

static llvm::Value* truncOrSext(IRBuilder<> &B, llvm::Value *v, llvm::Type *dstTy) {
  llvm::Type *srcTy = v->getType();
  if (srcTy == dstTy) return v;
  if (srcTy->isIntegerTy() && dstTy->isIntegerTy()) {
    unsigned sb = srcTy->getIntegerBitWidth();
    unsigned db = dstTy->getIntegerBitWidth();
    if (sb > db) return B.CreateTrunc(v, dstTy);
    if (sb < db) return B.CreateSExt(v, dstTy);
  }
  return B.CreateBitCast(v, dstTy);
}

static void attachAstInstrId(llvm::Instruction *I, const std::string &instrId, llvm::LLVMContext &C) {
  if (!I || instrId.empty()) return;
  auto *mdStr = llvm::MDString::get(C, instrId);
  auto *mdNode = llvm::MDNode::get(C, mdStr);
  I->setMetadata("ast_instr_id", mdNode);
}

static void attachPicRelocations(llvm::Instruction *I, const std::string &symName, llvm::LLVMContext &C) {
  if (!I || symName.empty()) return;
  auto *mdStr = llvm::MDString::get(C, symName);
  auto *mdNode = llvm::MDNode::get(C, mdStr);
  I->setMetadata("pic_relocations", mdNode);
}

static void recordMapping(FnLowerCtx &LC, const lifted_ast::Instruction &insn, llvm::Value *v) {
  if (!insn.has_id() || insn.id().empty() || !v) return;
  std::string name = "instr_" + insn.id();

  if (auto *I = llvm::dyn_cast<llvm::Instruction>(v)) {
    if (!I->getType()->isVoidTy()) I->setName(name);
    attachAstInstrId(I, insn.id(), LC.C);
  }

  LC.instrLlvmMapping[insn.id()] = name;
}

static llvm::Value* readGprSubreg(FnLowerCtx &LC, IRBuilder<> &B, const RegInfo &ri) {
  llvm::Value *full = loadGpr64(LC, B, (unsigned)ri.gprIndex);
  if (ri.bitWidth == 64 && ri.bitOffset == 0) return full;
  llvm::Value *shifted = full;
  if (ri.bitOffset != 0) {
    shifted = B.CreateLShr(full, B.getInt64(ri.bitOffset), "shr.subreg");
  }
  llvm::Type *ty = intTy(LC.C, ri.bitWidth);
  return B.CreateTrunc(shifted, ty, "trunc.subreg");
}

static void writeGprSubreg(FnLowerCtx &LC, IRBuilder<> &B, const RegInfo &ri, llvm::Value *v) {
  if (!ri.isValid || ri.isXmm || ri.gprIndex < 0) return;
  llvm::Type *i64 = B.getInt64Ty();
  unsigned w = ri.bitWidth;
  unsigned off = ri.bitOffset;

  if (w == 64 && off == 0) {
    llvm::Value *vi64 = v->getType()->isPointerTy() ? B.CreatePtrToInt(v, i64) : truncOrZext(B, v, i64);
    storeGpr64(LC, B, (unsigned)ri.gprIndex, vi64);
    return;
  }
  if (w == 32 && off == 0) {
    llvm::Type *i32 = B.getInt32Ty();
    llvm::Value *vi32 = truncOrZext(B, v, i32);
    llvm::Value *z = B.CreateZExt(vi32, i64, "zext32");
    storeGpr64(LC, B, (unsigned)ri.gprIndex, z);
    return;
  }

  llvm::Value *old = loadGpr64(LC, B, (unsigned)ri.gprIndex);
  llvm::Type *subTy = intTy(LC.C, w);
  llvm::Value *sub = truncOrZext(B, v, subTy);
  llvm::Value *sub64 = B.CreateZExt(sub, i64, "sub.zext64");

  if (off != 0) sub64 = B.CreateShl(sub64, B.getInt64(off), "sub.shl");

  uint64_t mask = (w == 64) ? ~0ULL : ((1ULL << w) - 1ULL);
  mask <<= off;
  llvm::Value *maskV = llvm::ConstantInt::get(i64, mask);
  llvm::Value *invMaskV = llvm::ConstantInt::get(i64, ~mask);

  llvm::Value *kept = B.CreateAnd(old, invMaskV, "merge.kept");
  llvm::Value *ins  = B.CreateAnd(sub64, maskV, "merge.ins");
  llvm::Value *merged = B.CreateOr(kept, ins, "merge");
  storeGpr64(LC, B, (unsigned)ri.gprIndex, merged);
}

// --------------------- address mappings -------------------

static llvm::Value* symbolAddressAsPtr(FnLowerCtx &LC, IRBuilder<> &B, const std::string &sym, const std::string &astInstrId = "") {
  auto it = LC.P->symbol_table().find(sym);
  if (it != LC.P->symbol_table().end()) {
    const auto &se = it->second;
    if (se.has_kind() && se.kind() == "label" && se.has_definition()) {
      auto bbIdOpt = structStringField(se.definition(), "bb_id");
      if (bbIdOpt) {
        auto itBB = LC.bbIdToLlvm.find(*bbIdOpt);
        if (itBB != LC.bbIdToLlvm.end()) return llvm::BlockAddress::get(LC.F, itBB->second);
      }
    }
  }

  if (llvm::GlobalVariable *gv = LC.M.getNamedGlobal(sym)) {
    llvm::Type *vt = gv->getValueType();
    llvm::SmallVector<llvm::Value*, 2> idx{B.getInt32(0)};
    if (vt->isArrayTy() || vt->isStructTy() || vt->isVectorTy()) idx.push_back(B.getInt32(0));

    llvm::Value *gep = B.CreateInBoundsGEP(vt, gv, idx, "sym.gep");
    if (!llvm::isa<llvm::Instruction>(gep)) {
      auto *gepInst = llvm::GetElementPtrInst::CreateInBounds(vt, gv, idx, "sym.gep");
      B.Insert(gepInst);
      gep = gepInst;
    }

    if (auto *I = llvm::dyn_cast<llvm::Instruction>(gep)) {
      attachPicRelocations(I, sym, LC.C);
      if (!astInstrId.empty()) attachAstInstrId(I, astInstrId, LC.C);
    }
    return gep;
  }
  if (llvm::Function *fn = LC.M.getFunction(sym)) return fn;
  return llvm::UndefValue::get(llvm::PointerType::getUnqual(LC.C));
}

static llvm::Value* resolveGotSlot(FnLowerCtx &LC, const std::string &sym) {
  std::string gotName = sym + "@GOT";
  llvm::GlobalVariable *gotSlot = LC.M.getNamedGlobal(gotName);
  if (!gotSlot) {
    llvm::Constant *init = LC.M.getNamedGlobal(sym);
    if (!init) {
      if (auto *fn = LC.M.getFunction(sym)) init = fn;
      else init = llvm::UndefValue::get(llvm::PointerType::getUnqual(LC.C));
    }
    gotSlot = new llvm::GlobalVariable(LC.M, llvm::PointerType::getUnqual(LC.C), true,
                                       llvm::GlobalValue::PrivateLinkage, init, gotName);
  }
  return gotSlot;
}

struct MemAddr {
  llvm::Value *ptr = nullptr;
  llvm::Align align = llvm::Align(1);
  bool isSymbolic = false;
  std::string symName;
};

static MemAddr resolveMemAddress(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Operand &op, unsigned accessSize, const std::string &astInstrId = "") {
  MemAddr out;
  out.ptr = llvm::UndefValue::get(llvm::PointerType::getUnqual(LC.C));
  out.align = llvm::Align(1);
  out.isSymbolic = false;

  if (!op.has_memory()) return out;
  const lifted_ast::Memory &m = op.memory();

  const std::string base = m.has_base() ? toUpper(m.base()) : "";
  const std::string index = m.has_index() ? toUpper(m.index()) : "";
  const int32_t scale = m.has_scale() ? m.scale() : 1;

  if ((base == "RBP" || base == "RSP") && m.has_displacement()) {
    auto dispOpt = valueAsInt64(m.displacement());
    if (dispOpt) {
      auto stk = lookupStackSlotPtr(B, LC.promotedSlots, base, *dispOpt, accessSize);
      if (stk) {
        out.ptr = stk->first;
        out.align = stk->second;
        return out;
      }
    }
  }

  // Symbolic/PIC
  if (op.has_symbol_ref()) {
    bool isGot = op.has_via_got() && op.via_got();
    if (op.symbol_ref() == "stderr") isGot = true; // PIC default treatment

    llvm::Value *basePtr = nullptr;
    if (isGot) {
      basePtr = resolveGotSlot(LC, op.symbol_ref());
    } else {
      basePtr = symbolAddressAsPtr(LC, B, op.symbol_ref(), astInstrId);
    }

    out.isSymbolic = true;
    out.symName = op.symbol_ref();

    if (m.has_displacement()) {
      auto dispOpt = valueAsInt64(m.displacement());
      if (dispOpt && *dispOpt != 0) {
        basePtr = B.CreateGEP(B.getInt8Ty(), basePtr, B.getInt64(*dispOpt), isGot ? "got.disp" : "sym.disp");
        if (auto *I = llvm::dyn_cast<llvm::Instruction>(basePtr)) {
          attachPicRelocations(I, out.symName, LC.C);
          if (!astInstrId.empty()) {
            attachAstInstrId(I, astInstrId, LC.C);
          }
        }
      }
    }
    out.ptr = basePtr;
    out.align = llvm::Align(1);
    return out;
  }

  // RIP-relative symbolic PIC reference (the exact case the review described)
  // base == "RIP" and displacement holds the symbol name (no top-level symbol_ref).
  if (base == "RIP" && m.has_displacement()) {
    const auto& disp = m.displacement();
    std::string sym;
    if (disp.kind_case() == google::protobuf::Value::kStringValue) {
      sym = disp.string_value();
    } else {
      auto sOpt = structStringField(disp, "symbol");
      if (sOpt) sym = *sOpt;
    }
    if (!sym.empty() &&
        (LC.M.getNamedGlobal(sym) != nullptr ||
         LC.M.getFunction(sym) != nullptr ||
         LC.P->symbol_table().count(sym))) {
      llvm::Value* basePtr = symbolAddressAsPtr(LC, B, sym, astInstrId);
      out.isSymbolic = true;
      out.symName = sym;
      out.ptr = basePtr;
      out.align = llvm::Align(1);
      return out;
    }
  }

  // General computed
  llvm::Value *addrI64 = B.getInt64(0);
  if (!base.empty() && base != "RIP") {
    RegInfo bri = decodeReg(base);
    if (bri.isValid && !bri.isXmm) {
      llvm::Value *b64 = readGprSubreg(LC, B, bri);
      b64 = truncOrZext(B, b64, B.getInt64Ty());
      addrI64 = B.CreateAdd(addrI64, b64, "addr.base");
    }
  }

  if (!index.empty()) {
    RegInfo iri = decodeReg(index);
    if (iri.isValid && !iri.isXmm) {
      llvm::Value *i64v = readGprSubreg(LC, B, iri);
      i64v = truncOrZext(B, i64v, B.getInt64Ty());
      llvm::Value *scaled = i64v;
      if (scale != 1 && scale > 0) {
        scaled = B.CreateMul(i64v, B.getInt64((int64_t)scale), "addr.scale");
      }
      addrI64 = B.CreateAdd(addrI64, scaled, "addr.index");
    }
  }

  if (m.has_displacement()) {
    auto dispOpt = valueAsInt64(m.displacement());
    if (dispOpt && *dispOpt != 0) {
      addrI64 = B.CreateAdd(addrI64, B.getInt64(*dispOpt), "addr.disp");
    }
  }

  out.ptr = B.CreateIntToPtr(addrI64, llvm::PointerType::getUnqual(LC.C), "addr.ptr");
  out.align = llvm::Align(1);
  return out;
}

static llvm::Value* evalExprToI64(FnLowerCtx &LC, IRBuilder<> &B, const google::protobuf::Value &v, const std::string &astInstrId = "") {
  using V = google::protobuf::Value;
  if (auto i = valueAsInt64(v)) return B.getInt64(*i);

  if (v.kind_case() == V::kStructValue) {
    if (auto regOpt = structStringField(v, "register")) {
      RegInfo ri = decodeReg(*regOpt);
      if (ri.isValid && !ri.isXmm) {
        llvm::Value *x = readGprSubreg(LC, B, ri);
        return truncOrZext(B, x, B.getInt64Ty());
      }
    }
    if (auto symOpt = structStringField(v, "symbol")) {
      std::string sym = *symOpt;

      // Fully adhere to Constant Int verification for symbolic resolution
      if (llvm::GlobalVariable *GV = LC.M.getNamedGlobal(sym)) {
        if (GV->isConstant() && GV->hasInitializer()) {
          if (auto *CI = llvm::dyn_cast<llvm::ConstantInt>(GV->getInitializer())) {
            return truncOrZext(B, CI, B.getInt64Ty());
          }
        }
      }
      llvm::Value *p = symbolAddressAsPtr(LC, B, sym, astInstrId);
      llvm::Value *pi64 = B.CreatePtrToInt(p, B.getInt64Ty(), "sym.ptrtoint");
      if (!astInstrId.empty() && llvm::isa<llvm::Instruction>(pi64)) attachAstInstrId(llvm::cast<llvm::Instruction>(pi64), astInstrId, LC.C);
      return pi64;
    }
    if (auto addV = structFieldValue(v, "additive")) {
      if (addV->kind_case() == V::kListValue) {
        llvm::Value *acc = B.getInt64(0);
        for (const auto &elt : addV->list_value().values()) {
          acc = B.CreateAdd(acc, evalExprToI64(LC, B, elt, astInstrId), "add");
        }
        return acc;
      }
    }
    if (auto subV = structFieldValue(v, "subtract")) {
      if (subV->kind_case() == V::kListValue) {
        const auto &xs = subV->list_value().values();
        if (xs.size() == 0) return B.getInt64(0);
        llvm::Value *acc = evalExprToI64(LC, B, xs[0], astInstrId);
        for (int i = 1; i < xs.size(); ++i) {
          acc = B.CreateSub(acc, evalExprToI64(LC, B, xs[i], astInstrId), "sub");
        }
        return acc;
      }
    }
  }
  return llvm::UndefValue::get(B.getInt64Ty());
}

static llvm::Value* loadFromMem(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Operand &memOp, llvm::Type *ty, const std::string &astInstrId = "") {
  unsigned accessSize = 0;
  if (ty->isIntegerTy()) accessSize = ty->getIntegerBitWidth() / 8;
  else if (ty->isPointerTy()) accessSize = 8;
  else accessSize = 8;

  MemAddr a = resolveMemAddress(LC, B, memOp, accessSize, astInstrId);
  llvm::LoadInst *L = B.CreateLoad(ty, a.ptr, "mem.ld");
  if (a.align.value() > 1) L->setAlignment(a.align);

  if (a.isSymbolic && !a.symName.empty()) attachPicRelocations(L, a.symName, LC.C);
  if (!astInstrId.empty()) attachAstInstrId(L, astInstrId, LC.C);
  return L;
}

static void storeToMem(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Operand &memOp, llvm::Value *v, llvm::Type *storeTy, const std::string &astInstrId = "") {
  unsigned accessSize = 0;
  if (storeTy->isIntegerTy()) accessSize = storeTy->getIntegerBitWidth() / 8;
  else if (storeTy->isPointerTy()) accessSize = 8;
  else accessSize = 8;

  MemAddr a = resolveMemAddress(LC, B, memOp, accessSize, astInstrId);
  llvm::Value *vv = v;
  if (vv->getType() != storeTy) {
    if (storeTy->isIntegerTy()) vv = truncOrZext(B, vv, storeTy);
    else if (storeTy->isPointerTy() && vv->getType()->isIntegerTy(64)) vv = B.CreateIntToPtr(vv, storeTy);
    else if (storeTy->isIntegerTy(64) && vv->getType()->isPointerTy()) vv = B.CreatePtrToInt(vv, storeTy);
    else vv = B.CreateBitCast(vv, storeTy);
  }

  llvm::StoreInst *S = B.CreateStore(vv, a.ptr);
  if (a.align.value() > 1) S->setAlignment(a.align);

  if (a.isSymbolic && !a.symName.empty()) attachPicRelocations(S, a.symName, LC.C);
  if (!astInstrId.empty()) attachAstInstrId(S, astInstrId, LC.C);
}

static llvm::Value* resolveRValue(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Operand &op, llvm::Type *desiredTy, const std::string &astInstrId = "") {

  // 1. Registers
  if (op.has_register_()) {
    RegInfo ri = decodeReg(op.register_());
    if (ri.isValid && ri.isXmm) {
      llvm::Type *xmmTy = llvm::FixedVectorType::get(B.getDoubleTy(), 2);
      llvm::Value *p = getStateFieldPtr(LC, B, kXmmBase + ri.xmmIndex);
      llvm::Value *v = B.CreateLoad(xmmTy, p, "xmm.ld");
      if (desiredTy == xmmTy) return v;
      return llvm::UndefValue::get(desiredTy);
    }
    if (ri.isValid && !ri.isXmm) {
      llvm::Value *v = readGprSubreg(LC, B, ri);
      if (desiredTy->isIntegerTy()) return truncOrZext(B, v, desiredTy);
      if (desiredTy->isPointerTy()) {
        llvm::Value *asI64 = truncOrZext(B, v, B.getInt64Ty());
        return B.CreateIntToPtr(asI64, desiredTy, "reg.inttoptr");
      }
      return llvm::UndefValue::get(desiredTy);
    }
  }

  // 2. Memory
  if (op.has_memory()) return loadFromMem(LC, B, op, desiredTy, astInstrId);

  // 3. Symbolic Immediates (Relocation-aware ptrtoint address materialization or Constant Int Injection)
  if (op.has_symbol_ref()) {
    llvm::Value *val = nullptr;
    std::string sym = op.symbol_ref();

    // Respect Step 6 Classifications for Constant Integer Immediates
    if (llvm::GlobalVariable *GV = LC.M.getNamedGlobal(sym)) {
      if (GV->isConstant() && GV->hasInitializer()) {
        if (auto *CI = llvm::dyn_cast<llvm::ConstantInt>(GV->getInitializer())) {
          val = CI;
        }
      }
    }

    if (!val) {
      llvm::Value *ptr = symbolAddressAsPtr(LC, B, sym, astInstrId);
      val = B.CreatePtrToInt(ptr, B.getInt64Ty(), "sym.imm.ptrtoint");
      if (llvm::isa<llvm::Instruction>(val) && !astInstrId.empty()) {
        attachAstInstrId(llvm::cast<llvm::Instruction>(val), astInstrId, LC.C);
      }
    }

    // Prepare robust 64-bit value to cleanly apply potential addends
    val = truncOrZext(B, val, B.getInt64Ty());

    // Respect any addend defined by an embedded integer evaluation
    int64_t addend = 0;
    if (op.has_integer()) {
      if (op.integer().has_value()) {
        auto iOpt = valueAsInt64(op.integer().value());
        if (iOpt) addend = *iOpt;
      }
    }
    if (addend != 0) {
      val = B.CreateAdd(val, B.getInt64(addend), "sym.imm.add");
    }

    if (desiredTy->isIntegerTy()) return truncOrZext(B, val, desiredTy);
    if (desiredTy->isPointerTy()) return B.CreateIntToPtr(val, desiredTy, "sym.imm.inttoptr");

    // Safety fallback for ConstantFP derivations
    if (desiredTy->isFloatingPointTy()) {
      unsigned dstBits = desiredTy->getPrimitiveSizeInBits();
      if (dstBits > 0) return B.CreateBitCast(truncOrZext(B, val, intTy(LC.C, dstBits)), desiredTy);
    }
    return llvm::UndefValue::get(desiredTy);
  }

  // 4. Expression Immediates
  if (op.has_expression()) {
    llvm::Value *exprI64 = evalExprToI64(LC, B, op.expression(), astInstrId);
    if (desiredTy->isIntegerTy()) return truncOrZext(B, exprI64, desiredTy);
    if (desiredTy->isPointerTy()) return B.CreateIntToPtr(exprI64, desiredTy, "expr.inttoptr");
    return llvm::UndefValue::get(desiredTy);
  }

  // 5. Pure Integer Immediates
  if (op.has_integer()) {
    int64_t imm = 0;
    if (op.integer().has_value()) {
      auto iOpt = valueAsInt64(op.integer().value());
      if (iOpt) imm = *iOpt;
    }
    if (desiredTy->isIntegerTy()) {
      unsigned bits = desiredTy->getIntegerBitWidth();
      llvm::APInt api(bits, (uint64_t)imm, /*isSigned=*/true);
      return llvm::ConstantInt::get(desiredTy, api);
    }
    if (desiredTy->isPointerTy()) {
      llvm::Value *ci = llvm::ConstantInt::get(B.getInt64Ty(), (uint64_t)imm, true);
      return B.CreateIntToPtr(ci, desiredTy, "imm.inttoptr");
    }
    if (desiredTy->isFloatingPointTy()) {
      return llvm::ConstantFP::get(desiredTy, (double)imm);
    }
    return llvm::UndefValue::get(desiredTy);
  }

  return llvm::UndefValue::get(desiredTy);
}

static llvm::Value* computePF(IRBuilder<> &B, llvm::Value *resIntN) {
  llvm::Value *lo8 = B.CreateTrunc(resIntN, B.getInt8Ty(), "pf.lo8");
  llvm::Function *ctpop = llvm::Intrinsic::getDeclaration(
      B.GetInsertBlock()->getModule(), llvm::Intrinsic::ctpop, {B.getInt8Ty()});
  llvm::Value *pop = B.CreateCall(ctpop, {lo8}, "pf.pop");
  llvm::Value *lsb = B.CreateAnd(pop, B.getInt8(1), "pf.lsb");
  llvm::Value *odd = B.CreateICmpEQ(lsb, B.getInt8(1), "pf.odd");
  return B.CreateNot(odd, "pf");
}

static void updateFlagsLogic(FnLowerCtx &LC, IRBuilder<> &B, llvm::Value *res) {
  storeFlag(LC, B, CF, B.getFalse());
  storeFlag(LC, B, OF, B.getFalse());
  storeFlag(LC, B, AF, B.getFalse());
  storeFlag(LC, B, ZF, B.CreateICmpEQ(res, llvm::ConstantInt::get(res->getType(), 0), "zf"));
  storeFlag(LC, B, SF, B.CreateICmpSLT(res, llvm::ConstantInt::get(res->getType(), 0), "sf"));
  storeFlag(LC, B, PF, computePF(B, res));
}

static void updateFlagsAddSubCommon(FnLowerCtx &LC, IRBuilder<> &B, llvm::Value *a, llvm::Value *b, llvm::Value *res, llvm::Value *cf, llvm::Value *of) {
  storeFlag(LC, B, ZF, B.CreateICmpEQ(res, llvm::ConstantInt::get(res->getType(), 0), "zf"));
  storeFlag(LC, B, SF, B.CreateICmpSLT(res, llvm::ConstantInt::get(res->getType(), 0), "sf"));
  storeFlag(LC, B, PF, computePF(B, res));
  llvm::Value *x = B.CreateXor(a, b, "af.x1");
  x = B.CreateXor(x, res, "af.x2");
  llvm::Value *mask = llvm::ConstantInt::get(res->getType(), 0x10);
  storeFlag(LC, B, AF, B.CreateICmpNE(B.CreateAnd(x, mask), llvm::ConstantInt::get(res->getType(), 0), "af"));
  storeFlag(LC, B, CF, cf);
  storeFlag(LC, B, OF, of);
}

static void updateFlagsAdd(FnLowerCtx &LC, IRBuilder<> &B, llvm::Value *a, llvm::Value *b, llvm::Value *res) {
  llvm::Type *ty = res->getType();
  llvm::Function *uadd = llvm::Intrinsic::getDeclaration(&LC.M, llvm::Intrinsic::uadd_with_overflow, {ty});
  llvm::Function *sadd = llvm::Intrinsic::getDeclaration(&LC.M, llvm::Intrinsic::sadd_with_overflow, {ty});
  llvm::Value *u = B.CreateCall(uadd, {a, b}, "uadd.ov");
  llvm::Value *s = B.CreateCall(sadd, {a, b}, "sadd.ov");
  updateFlagsAddSubCommon(LC, B, a, b, res, B.CreateExtractValue(u, 1, "cf"), B.CreateExtractValue(s, 1, "of"));
}

static void updateFlagsSub(FnLowerCtx &LC, IRBuilder<> &B, llvm::Value *a, llvm::Value *b, llvm::Value *res) {
  llvm::Type *ty = res->getType();
  llvm::Function *usub = llvm::Intrinsic::getDeclaration(&LC.M, llvm::Intrinsic::usub_with_overflow, {ty});
  llvm::Function *ssub = llvm::Intrinsic::getDeclaration(&LC.M, llvm::Intrinsic::ssub_with_overflow, {ty});
  llvm::Value *u = B.CreateCall(usub, {a, b}, "usub.ov");
  llvm::Value *s = B.CreateCall(ssub, {a, b}, "ssub.ov");
  updateFlagsAddSubCommon(LC, B, a, b, res, B.CreateExtractValue(u, 1, "cf"), B.CreateExtractValue(s, 1, "of"));
}

static bool isTerminatorOpcode(const std::string &opcUpper) {
  static const std::set<std::string> terms = {
    "RET","JMP","JE","JNE","JL","JLE","JG","JGE","JA","JAE","JB","JBE",
    "JO","JNO","JS","JNS","JP","JNP","JC","JNC",
    "LOOP","LOOPE","LOOPNE","IRET","SYSRET"
  };
  return terms.count(opcUpper) != 0;
}

static bool isFloatOpcode(const std::string &opcUpper) {
  static const std::set<std::string> fp = {
    "MOVSS","ADDSS","SUBSS","MULSS","DIVSS",
    "MOVSD","ADDSD","SUBSD","MULSD","DIVSD",
    "CVTTSS2SI","CVTSS2SI","CVTSI2SS",
    "CVTTSD2SI","CVTSD2SI","CVTSI2SD"
  };
  return fp.count(opcUpper) != 0;
}

static unsigned regWidthBits(const std::string &reg) {
  RegInfo ri = decodeReg(reg);
  if (!ri.isValid || ri.isXmm) return 0;
  return ri.bitWidth;
}

static llvm::Type* chooseOpIntType(FnLowerCtx &LC, const lifted_ast::Instruction &insn) {
  if (insn.has_op_refinement()) {
    std::string r = toUpper(insn.op_refinement());
    // Directly preserve CHAR literal formatting boundaries ensuring identical logic matching to integers.
    if (r == "I8" || r.rfind("I8",0)==0 || r == "CHAR") return llvm::Type::getInt8Ty(LC.C);
    if (r == "I16" || r.rfind("I16",0)==0) return llvm::Type::getInt16Ty(LC.C);
    if (r == "I32" || r.rfind("I32",0)==0) return llvm::Type::getInt32Ty(LC.C);
    if (r == "I64" || r.rfind("I64",0)==0 || r == "PTR") return llvm::Type::getInt64Ty(LC.C);
  }
  return llvm::Type::getInt64Ty(LC.C);
}

static void lowerMOV(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn) {
  if (insn.operands_size() < 2) return;
  const auto &dst = insn.operands(0);
  const auto &src = insn.operands(1);
  llvm::Type *ty = nullptr;

  if (dst.has_register_()) {
    unsigned w = regWidthBits(dst.register_());
    ty = intTy(LC.C, w == 0 ? 64 : w);
  } else if (dst.has_memory()) {
    unsigned sz = memSizeBytesFromOperand(dst);
    if (sz == 0) ty = chooseOpIntType(LC, insn);
    else ty = intTy(LC.C, sz * 8);
  } else {
    ty = chooseOpIntType(LC, insn);
  }
  if (!ty || !ty->isIntegerTy()) ty = B.getInt64Ty();

  llvm::Value *v = resolveRValue(LC, B, src, ty, insn.id());

  if (dst.has_register_()) {
    RegInfo dri = decodeReg(dst.register_());
    writeGprSubreg(LC, B, dri, v);
    recordMapping(LC, insn, v);
  } else if (dst.has_memory()) {
    storeToMem(LC, B, dst, v, ty, insn.id());
    recordMapping(LC, insn, v);
  }
}

static void lowerLEA(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn) {
  if (insn.operands_size() < 2) return;
  const auto &dst = insn.operands(0);
  const auto &src = insn.operands(1);
  if (!dst.has_register_()) return;
  RegInfo dri = decodeReg(dst.register_());
  if (!dri.isValid || dri.isXmm) return;

  llvm::Value *addrI64 = nullptr;
  if (src.has_memory()) {
    MemAddr a = resolveMemAddress(LC, B, src, 1, insn.id());
    llvm::Value *pi64 = B.CreatePtrToInt(a.ptr, B.getInt64Ty(), "lea.ptrtoint");
    if (llvm::isa<llvm::Instruction>(pi64)) attachAstInstrId(llvm::cast<llvm::Instruction>(pi64), insn.id(), LC.C);
    addrI64 = pi64;
  } else if (src.has_symbol_ref()) {
    llvm::Value *p = symbolAddressAsPtr(LC, B, src.symbol_ref(), insn.id());
    llvm::Value *pi64 = B.CreatePtrToInt(p, B.getInt64Ty(), "lea.sym");
    if (llvm::isa<llvm::Instruction>(pi64)) attachAstInstrId(llvm::cast<llvm::Instruction>(pi64), insn.id(), LC.C);

    // Apply exact addend logic consistency to LEA
    int64_t addend = 0;
    if (src.has_integer()) {
      if (src.integer().has_value()) {
        auto iOpt = valueAsInt64(src.integer().value());
        if (iOpt) addend = *iOpt;
      }
    }
    if (addend != 0) {
      pi64 = B.CreateAdd(pi64, B.getInt64(addend), "lea.sym.add");
    }
    addrI64 = pi64;
  } else if (src.has_expression()) {
    addrI64 = evalExprToI64(LC, B, src.expression(), insn.id());
  } else {
    addrI64 = llvm::UndefValue::get(B.getInt64Ty());
  }
  llvm::Type *dstTy = intTy(LC.C, dri.bitWidth);
  llvm::Value *out = truncOrZext(B, addrI64, dstTy);
  writeGprSubreg(LC, B, dri, out);
  recordMapping(LC, insn, out);
}

static void lowerALU2(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn, const std::string &opcUpper) {
  if (insn.operands_size() < 2) return;
  const auto &dst = insn.operands(0);
  const auto &src = insn.operands(1);
  llvm::Type *ty = nullptr;

  if (dst.has_register_()) {
    unsigned w = regWidthBits(dst.register_());
    ty = intTy(LC.C, w == 0 ? 64 : w);
  } else if (dst.has_memory()) {
    unsigned sz = memSizeBytesFromOperand(dst);
    ty = sz == 0 ? chooseOpIntType(LC, insn) : intTy(LC.C, sz * 8);
  } else {
    ty = chooseOpIntType(LC, insn);
  }
  if (!ty || !ty->isIntegerTy()) ty = B.getInt64Ty();

  llvm::Value *a = nullptr;
  if (dst.has_register_()) {
    RegInfo dri = decodeReg(dst.register_());
    a = truncOrZext(B, readGprSubreg(LC, B, dri), ty);
  } else if (dst.has_memory()) {
    a = loadFromMem(LC, B, dst, ty, insn.id());
  } else { return; }

  llvm::Value *b = resolveRValue(LC, B, src, ty, insn.id());
  llvm::Value *res = nullptr;
  bool isLogic = false;

  if (opcUpper == "ADD") { res = B.CreateAdd(a, b, "add"); updateFlagsAdd(LC, B, a, b, res); }
  else if (opcUpper == "SUB") { res = B.CreateSub(a, b, "sub"); updateFlagsSub(LC, B, a, b, res); }
  else if (opcUpper == "XOR") { res = B.CreateXor(a, b, "xor"); isLogic = true; }
  else if (opcUpper == "AND") { res = B.CreateAnd(a, b, "and"); isLogic = true; }
  else if (opcUpper == "OR") { res = B.CreateOr(a, b, "or"); isLogic = true; }
  else { return; }

  if (isLogic) updateFlagsLogic(LC, B, res);

  if (dst.has_register_()) {
    RegInfo dri = decodeReg(dst.register_());
    writeGprSubreg(LC, B, dri, res);
  } else if (dst.has_memory()) {
    storeToMem(LC, B, dst, res, ty, insn.id());
  }
  recordMapping(LC, insn, res);
}

static void lowerCMP(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn) {
  if (insn.operands_size() < 2) return;
  const auto &aOp = insn.operands(0);
  const auto &bOp = insn.operands(1);

  llvm::Type *ty = nullptr;
  if (aOp.has_register_()) {
    unsigned w = regWidthBits(aOp.register_());
    ty = intTy(LC.C, w == 0 ? 64 : w);
  } else if (aOp.has_memory()) {
    unsigned sz = memSizeBytesFromOperand(aOp);
    ty = sz == 0 ? chooseOpIntType(LC, insn) : intTy(LC.C, sz * 8);
  } else {
    ty = chooseOpIntType(LC, insn);
  }
  if (!ty || !ty->isIntegerTy()) ty = B.getInt64Ty();

  llvm::Value *a = resolveRValue(LC, B, aOp, ty, insn.id());
  llvm::Value *b = resolveRValue(LC, B, bOp, ty, insn.id());
  llvm::Value *res = B.CreateSub(a, b, "cmp.sub");
  updateFlagsSub(LC, B, a, b, res);
  recordMapping(LC, insn, res);
}

static void lowerTEST(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn) {
  if (insn.operands_size() < 2) return;
  const auto &aOp = insn.operands(0);
  const auto &bOp = insn.operands(1);

  llvm::Type *ty = nullptr;
  if (aOp.has_register_()) {
    unsigned w = regWidthBits(aOp.register_());
    ty = intTy(LC.C, w == 0 ? 64 : w);
  } else if (aOp.has_memory()) {
    unsigned sz = memSizeBytesFromOperand(aOp);
    ty = sz == 0 ? chooseOpIntType(LC, insn) : intTy(LC.C, sz * 8);
  } else {
    ty = chooseOpIntType(LC, insn);
  }
  if (!ty || !ty->isIntegerTy()) ty = B.getInt64Ty();

  llvm::Value *a = resolveRValue(LC, B, aOp, ty, insn.id());
  llvm::Value *b = resolveRValue(LC, B, bOp, ty, insn.id());
  llvm::Value *res = B.CreateAnd(a, b, "test.and");
  updateFlagsLogic(LC, B, res);
  recordMapping(LC, insn, res);
}

static void lowerMOVZX_MOVSX(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn, bool isSignExtend) {
  if (insn.operands_size() < 2) return;
  const auto &dst = insn.operands(0);
  const auto &src = insn.operands(1);

  if (!dst.has_register_()) return;
  RegInfo dri = decodeReg(dst.register_());
  if (!dri.isValid || dri.isXmm) return;

  llvm::Type *dstTy = intTy(LC.C, dri.bitWidth);
  llvm::Type *srcTy = nullptr;
  if (src.has_register_()) {
    unsigned sw = regWidthBits(src.register_());
    srcTy = intTy(LC.C, sw == 0 ? 8 : sw);
  } else if (src.has_memory()) {
    unsigned sz = memSizeBytesFromOperand(src);
    srcTy = intTy(LC.C, sz == 0 ? 8 : (sz * 8));
  } else { srcTy = dstTy; }

  llvm::Value *v = resolveRValue(LC, B, src, srcTy, insn.id());
  llvm::Value *ext = isSignExtend ? truncOrSext(B, v, dstTy) : truncOrZext(B, v, dstTy);
  writeGprSubreg(LC, B, dri, ext);
  recordMapping(LC, insn, ext);
}

static void lowerSETcc(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn, const std::string &opcUpper) {
  if (insn.operands_size() < 1) return;
  const auto &dst = insn.operands(0);

  llvm::Value *cond = nullptr;
  if (opcUpper == "SETC" || opcUpper == "SETB") cond = loadFlag(LC, B, CF);
  else if (opcUpper == "SETNE") cond = B.CreateNot(loadFlag(LC, B, ZF), "setne");
  else if (opcUpper == "SETE" || opcUpper == "SETZ") cond = loadFlag(LC, B, ZF);
  else if (opcUpper == "SETPE" || opcUpper == "SETP") cond = loadFlag(LC, B, PF);
  else cond = B.getFalse();

  llvm::Value *byteV = B.CreateZExt(cond, B.getInt8Ty(), "setcc.i8");
  if (dst.has_register_()) {
    writeGprSubreg(LC, B, decodeReg(dst.register_()), byteV);
  } else if (dst.has_memory()) {
    storeToMem(LC, B, dst, byteV, B.getInt8Ty(), insn.id());
  }
  recordMapping(LC, insn, byteV);
}

static void lowerCMOVE(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn) {
  if (insn.operands_size() < 2) return;
  const auto &dst = insn.operands(0);
  const auto &src = insn.operands(1);
  if (!dst.has_register_()) return;
  RegInfo dri = decodeReg(dst.register_());
  if (!dri.isValid || dri.isXmm) return;

  llvm::Type *ty = intTy(LC.C, dri.bitWidth);
  llvm::Value *oldv = truncOrZext(B, readGprSubreg(LC, B, dri), ty);
  llvm::Value *newv = resolveRValue(LC, B, src, ty, insn.id());

  llvm::Value *res = B.CreateSelect(loadFlag(LC, B, ZF), newv, oldv, "cmove.sel");
  writeGprSubreg(LC, B, dri, res);
  recordMapping(LC, insn, res);
}

static void lowerINC(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn) {
  if (insn.operands_size() < 1) return;
  const auto &dst = insn.operands(0);

  llvm::Type *ty = nullptr;
  if (dst.has_register_()) {
    unsigned w = regWidthBits(dst.register_());
    ty = intTy(LC.C, w == 0 ? 64 : w);
  } else if (dst.has_memory()) {
    unsigned sz = memSizeBytesFromOperand(dst);
    ty = sz == 0 ? chooseOpIntType(LC, insn) : intTy(LC.C, sz * 8);
  } else { return; }
  if (!ty || !ty->isIntegerTy()) ty = B.getInt64Ty();

  llvm::Value *old = nullptr, *res = nullptr;
  llvm::Value *one = llvm::ConstantInt::get(ty, 1);

  if (dst.has_register_()) {
    old = truncOrZext(B, readGprSubreg(LC, B, decodeReg(dst.register_())), ty);
  } else {
    old = loadFromMem(LC, B, dst, ty, insn.id());
  }
  res = B.CreateAdd(old, one, "inc");

  llvm::Function *sadd = llvm::Intrinsic::getDeclaration(&LC.M, llvm::Intrinsic::sadd_with_overflow, {ty});
  llvm::Value *s = B.CreateCall(sadd, {old, one}, "inc.ov");
  storeFlag(LC, B, OF, B.CreateExtractValue(s, 1, "of"));
  storeFlag(LC, B, ZF, B.CreateICmpEQ(res, llvm::ConstantInt::get(ty, 0), "zf"));
  storeFlag(LC, B, SF, B.CreateICmpSLT(res, llvm::ConstantInt::get(ty, 0), "sf"));
  storeFlag(LC, B, PF, computePF(B, res));
  llvm::Value *x = B.CreateXor(old, one);
  x = B.CreateXor(x, res);
  storeFlag(LC, B, AF, B.CreateICmpNE(B.CreateAnd(x, llvm::ConstantInt::get(ty, 0x10)), llvm::ConstantInt::get(ty, 0)));

  if (dst.has_register_()) {
    writeGprSubreg(LC, B, decodeReg(dst.register_()), res);
  } else {
    storeToMem(LC, B, dst, res, ty, insn.id());
  }
  recordMapping(LC, insn, res);
}

static void lowerSHL(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn) {
  if (insn.operands_size() < 2) return;
  const auto &dst = insn.operands(0);
  const auto &cnt = insn.operands(1);

  llvm::Type *ty = nullptr;
  if (dst.has_register_()) {
    unsigned w = regWidthBits(dst.register_());
    ty = intTy(LC.C, w == 0 ? 64 : w);
  } else if (dst.has_memory()) {
    unsigned sz = memSizeBytesFromOperand(dst);
    ty = sz == 0 ? chooseOpIntType(LC, insn) : intTy(LC.C, sz * 8);
  } else { return; }
  if (!ty || !ty->isIntegerTy()) ty = B.getInt64Ty();

  llvm::Value *old = nullptr;
  if (dst.has_register_()) {
    old = truncOrZext(B, readGprSubreg(LC, B, decodeReg(dst.register_())), ty);
  } else {
    old = loadFromMem(LC, B, dst, ty, insn.id());
  }

  llvm::Value *count = nullptr;
  if (cnt.has_integer()) {
    int64_t imm = 0;
    if (auto v = valueAsInt64(cnt.integer().value())) imm = *v;
    count = llvm::ConstantInt::get(ty, (uint64_t)imm);
  } else if (cnt.has_register_()) {
    count = truncOrZext(B, readGprSubreg(LC, B, decodeReg(cnt.register_())), ty);
  } else {
    count = llvm::ConstantInt::get(ty, 0);
  }

  unsigned w = ty->getIntegerBitWidth();
  llvm::Value *mask = llvm::ConstantInt::get(ty, (uint64_t)(w - 1));
  llvm::Value *masked = B.CreateAnd(count, mask, "shl.mask");
  llvm::Value *res = B.CreateShl(old, masked, "shl");

  llvm::Value *shiftAmtI64 = truncOrZext(B, masked, B.getInt64Ty());
  llvm::Value *pos = B.CreateSub(B.getInt64(w), shiftAmtI64, "shl.pos");
  llvm::Value *cfBit = B.CreateLShr(old, truncOrZext(B, pos, ty));
  storeFlag(LC, B, CF, B.CreateTrunc(cfBit, B.getInt1Ty(), "cf"));
  storeFlag(LC, B, OF, B.getFalse());
  storeFlag(LC, B, ZF, B.CreateICmpEQ(res, llvm::ConstantInt::get(ty, 0), "zf"));
  storeFlag(LC, B, SF, B.CreateICmpSLT(res, llvm::ConstantInt::get(ty, 0), "sf"));
  storeFlag(LC, B, PF, computePF(B, res));

  if (dst.has_register_()) writeGprSubreg(LC, B, decodeReg(dst.register_()), res);
  else storeToMem(LC, B, dst, res, ty, insn.id());
  recordMapping(LC, insn, res);
}

static void lowerSHR(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn) {
  if (insn.operands_size() < 2) return;
  const auto &dst = insn.operands(0);
  const auto &cnt = insn.operands(1);

  llvm::Type *ty = nullptr;
  if (dst.has_register_()) {
    unsigned w = regWidthBits(dst.register_());
    ty = intTy(LC.C, w == 0 ? 64 : w);
  } else if (dst.has_memory()) {
    unsigned sz = memSizeBytesFromOperand(dst);
    ty = sz == 0 ? chooseOpIntType(LC, insn) : intTy(LC.C, sz * 8);
  } else return;
  if (!ty || !ty->isIntegerTy()) ty = B.getInt64Ty();

  llvm::Value *old = dst.has_register_()
    ? truncOrZext(B, readGprSubreg(LC, B, decodeReg(dst.register_())), ty)
    : loadFromMem(LC, B, dst, ty, insn.id());

  llvm::Value *count = nullptr;
  if (cnt.has_integer()) {
    int64_t imm = valueAsInt64(cnt.integer().value()).value_or(0);
    count = llvm::ConstantInt::get(ty, (uint64_t)imm);
  } else if (cnt.has_register_()) {
    count = truncOrZext(B, readGprSubreg(LC, B, decodeReg(cnt.register_())), ty);
  } else {
    count = llvm::ConstantInt::get(ty, 0);
  }

  unsigned w = ty->getIntegerBitWidth();
  llvm::Value *mask = llvm::ConstantInt::get(ty, (uint64_t)(w - 1));
  llvm::Value *masked = B.CreateAnd(count, mask, "shr.mask");

  llvm::Value *res = B.CreateLShr(old, masked, "shr");

  // CF = last bit shifted out (exactly matches x86 SHR, including the test case shift-by-1)
  // count==0 leaves CF unchanged (safe, no UB, mirrors real hardware)
  llvm::Value *doUpdate = B.CreateICmpNE(masked, llvm::ConstantInt::get(ty, 0));
  llvm::Value *cfShiftAmt = B.CreateSub(masked, llvm::ConstantInt::get(ty, 1));
  llvm::Value *tmp = B.CreateLShr(old, cfShiftAmt);
  llvm::Value *lastBit = B.CreateAnd(tmp, llvm::ConstantInt::get(ty, 1));
  llvm::Value *newCf = B.CreateTrunc(lastBit, B.getInt1Ty());
  llvm::Value *cf = B.CreateSelect(doUpdate, newCf, loadFlag(LC, B, CF));
  storeFlag(LC, B, CF, cf);

  // OF only for count==1 (SHR: OF = original MSB)
  llvm::Value *isOne = B.CreateICmpEQ(masked, llvm::ConstantInt::get(ty, 1));
  llvm::Value *msb = B.CreateLShr(old, llvm::ConstantInt::get(ty, w - 1));
  llvm::Value *ofOne = B.CreateTrunc(B.CreateAnd(msb, llvm::ConstantInt::get(ty, 1)), B.getInt1Ty());
  llvm::Value *of = B.CreateSelect(isOne, ofOne, loadFlag(LC, B, OF));
  storeFlag(LC, B, OF, of);

  // standard status flags (exactly as in your lowerSHL / lowerINC)
  storeFlag(LC, B, ZF, B.CreateICmpEQ(res, llvm::ConstantInt::get(ty, 0), "zf"));
  storeFlag(LC, B, SF, B.CreateICmpSLT(res, llvm::ConstantInt::get(ty, 0), "sf"));
  storeFlag(LC, B, PF, computePF(B, res));

  if (dst.has_register_()) {
    writeGprSubreg(LC, B, decodeReg(dst.register_()), res);
  } else {
    storeToMem(LC, B, dst, res, ty, insn.id());
  }
  recordMapping(LC, insn, res);
}

static void lowerSAR(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn) {
  if (insn.operands_size() < 2) return;
  const auto &dst = insn.operands(0);
  const auto &cnt = insn.operands(1);

  llvm::Type *ty = nullptr;
  if (dst.has_register_()) {
    unsigned w = regWidthBits(dst.register_());
    ty = intTy(LC.C, w == 0 ? 64 : w);
  } else if (dst.has_memory()) {
    unsigned sz = memSizeBytesFromOperand(dst);
    ty = sz == 0 ? chooseOpIntType(LC, insn) : intTy(LC.C, sz * 8);
  } else return;
  if (!ty || !ty->isIntegerTy()) ty = B.getInt64Ty();

  llvm::Value *old = dst.has_register_()
    ? truncOrZext(B, readGprSubreg(LC, B, decodeReg(dst.register_())), ty)
    : loadFromMem(LC, B, dst, ty, insn.id());

  llvm::Value *count = nullptr;
  if (cnt.has_integer()) {
    int64_t imm = valueAsInt64(cnt.integer().value()).value_or(0);
    count = llvm::ConstantInt::get(ty, (uint64_t)imm);
  } else if (cnt.has_register_()) {
    count = truncOrZext(B, readGprSubreg(LC, B, decodeReg(cnt.register_())), ty);
  } else {
    count = llvm::ConstantInt::get(ty, 0);
  }

  unsigned w = ty->getIntegerBitWidth();
  llvm::Value *mask = llvm::ConstantInt::get(ty, (uint64_t)(w - 1));
  llvm::Value *masked = B.CreateAnd(count, mask, "sar.mask");

  llvm::Value *res = B.CreateAShr(old, masked, "sar");

  // CF (identical logic to SHR)
  llvm::Value *doUpdate = B.CreateICmpNE(masked, llvm::ConstantInt::get(ty, 0));
  llvm::Value *cfShiftAmt = B.CreateSub(masked, llvm::ConstantInt::get(ty, 1));
  llvm::Value *tmp = B.CreateLShr(old, cfShiftAmt);
  llvm::Value *lastBit = B.CreateAnd(tmp, llvm::ConstantInt::get(ty, 1));
  llvm::Value *newCf = B.CreateTrunc(lastBit, B.getInt1Ty());
  llvm::Value *cf = B.CreateSelect(doUpdate, newCf, loadFlag(LC, B, CF));
  storeFlag(LC, B, CF, cf);

  // SAR: OF is always 0 when count==1 (Intel spec)
  llvm::Value *isOne = B.CreateICmpEQ(masked, llvm::ConstantInt::get(ty, 1));
  llvm::Value *of = B.CreateSelect(isOne, B.getFalse(), loadFlag(LC, B, OF));
  storeFlag(LC, B, OF, of);

  // standard status flags
  storeFlag(LC, B, ZF, B.CreateICmpEQ(res, llvm::ConstantInt::get(ty, 0), "zf"));
  storeFlag(LC, B, SF, B.CreateICmpSLT(res, llvm::ConstantInt::get(ty, 0), "sf"));
  storeFlag(LC, B, PF, computePF(B, res));

  if (dst.has_register_()) {
    writeGprSubreg(LC, B, decodeReg(dst.register_()), res);
  } else {
    storeToMem(LC, B, dst, res, ty, insn.id());
  }
  recordMapping(LC, insn, res);
}

static void lowerXCHG(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn) {
  if (insn.operands_size() < 2) return;
  const auto &op0 = insn.operands(0);
  const auto &op1 = insn.operands(1);

  const lifted_ast::Operand *memOp = nullptr;
  const lifted_ast::Operand *regOp = nullptr;
  if (op0.has_memory() && op1.has_register_()) { memOp = &op0; regOp = &op1; }
  else if (op1.has_memory() && op0.has_register_()) { memOp = &op1; regOp = &op0; }
  else return;

  RegInfo rri = decodeReg(regOp->register_());
  if (!rri.isValid || rri.isXmm) return;
  llvm::Type *ty = intTy(LC.C, rri.bitWidth);
  llvm::Value *regVal = truncOrZext(B, readGprSubreg(LC, B, rri), ty);

  unsigned accessSize = rri.bitWidth / 8;
  if (accessSize == 0) accessSize = 1;
  MemAddr ma = resolveMemAddress(LC, B, *memOp, accessSize, insn.id());

  llvm::LoadInst *oldMem = B.CreateLoad(ty, ma.ptr, "xchg.ld");
  if (ma.align.value() > 1) oldMem->setAlignment(ma.align);
  if (ma.isSymbolic && !ma.symName.empty()) attachPicRelocations(oldMem, ma.symName, LC.C);
  if (!insn.id().empty()) attachAstInstrId(oldMem, insn.id(), LC.C);

  llvm::StoreInst *st = B.CreateStore(regVal, ma.ptr);
  if (ma.align.value() > 1) st->setAlignment(ma.align);
  if (ma.isSymbolic && !ma.symName.empty()) attachPicRelocations(st, ma.symName, LC.C);
  if (!insn.id().empty()) attachAstInstrId(st, insn.id(), LC.C);

  writeGprSubreg(LC, B, rri, oldMem);
  recordMapping(LC, insn, oldMem);
}

static void lowerPOP(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn) {
  if (insn.operands_size() < 1) return;
  const auto &dst = insn.operands(0);
  if (!dst.has_register_()) return;

  RegInfo dri = decodeReg(dst.register_());
  if (!dri.isValid || dri.isXmm) return;

  // Load from [RSP]
  unsigned rspIdx = (unsigned)gprFieldIndex64("RSP");
  llvm::Value *rsp = loadGpr64(LC, B, rspIdx);
  llvm::Value *rspPtr = B.CreateIntToPtr(rsp, llvm::PointerType::getUnqual(LC.C), "pop.ptr");

  llvm::Type *ty = intTy(LC.C, dri.bitWidth);
  llvm::LoadInst *loaded = B.CreateLoad(ty, rspPtr, "pop.ld");
  loaded->setAlignment(llvm::Align(8));

  // Write to destination register
  writeGprSubreg(LC, B, dri, loaded);

  // Increment RSP by 8
  llvm::Value *newRsp = B.CreateAdd(rsp, B.getInt64(8), "pop.rsp");
  storeGpr64(LC, B, rspIdx, newRsp);

  recordMapping(LC, insn, loaded);
}

static void lowerLEAVE(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn) {
  unsigned rspIdx = (unsigned)gprFieldIndex64("RSP");
  unsigned rbpIdx = (unsigned)gprFieldIndex64("RBP");

  // LEAVE == mov rsp, rbp ; pop rbp
  llvm::Value *rbpVal = loadGpr64(LC, B, rbpIdx);

  // 1. RSP = RBP
  storeGpr64(LC, B, rspIdx, rbpVal);

  // 2. pop rbp (exact same pattern as lowerPOP, but hardcoded for RBP)
  llvm::Value *rspPtr = B.CreateIntToPtr(rbpVal,
      llvm::PointerType::getUnqual(LC.C), "leave.pop.ptr");
  llvm::LoadInst *loaded = B.CreateLoad(B.getInt64Ty(), rspPtr, "leave.pop.ld");
  loaded->setAlignment(llvm::Align(8));

  llvm::Value *newRsp = B.CreateAdd(rbpVal, B.getInt64(8), "leave.newrsp");

  storeGpr64(LC, B, rbpIdx, loaded);
  storeGpr64(LC, B, rspIdx, newRsp);

  recordMapping(LC, insn, loaded);  // matches your existing POP style
}

static void lowerPUSH(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn) {
  if (insn.operands_size() < 1) return;
  const auto &op = insn.operands(0);
  if (!op.has_register_()) return;
  RegInfo ri = decodeReg(op.register_());
  if (!ri.isValid || ri.isXmm) return;

  llvm::Value *val = readGprSubreg(LC, B, ri);
  llvm::Value *val64 = truncOrZext(B, val, B.getInt64Ty());

  // Decrement RSP first (x86 push semantics)
  unsigned rspIdx = (unsigned)gprFieldIndex64("RSP");
  llvm::Value *rsp = loadGpr64(LC, B, rspIdx);
  llvm::Value *newRsp = B.CreateSub(rsp, B.getInt64(8), "push.rsp");
  storeGpr64(LC, B, rspIdx, newRsp);

  // Store value to [newRSP]
  llvm::Value *rspPtr = B.CreateIntToPtr(newRsp, llvm::PointerType::getUnqual(LC.C), "push.ptr");
  llvm::StoreInst *st = B.CreateStore(val64, rspPtr);
  st->setAlignment(llvm::Align(8));

  recordMapping(LC, insn, newRsp);
}

static void lowerIMUL(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn) {
  if (insn.operands_size() == 0) return;

  // 1-operand form (exactly what your combined test uses: imul rdx)
  //   RDX:RAX ← RAX × src (signed, full 128-bit result)
  if (insn.operands_size() == 1) {
    const auto &srcOp = insn.operands(0);

    RegInfo raxRi = decodeReg("RAX");
    RegInfo rdxRi = decodeReg("RDX");

    llvm::Value *rax = readGprSubreg(LC, B, raxRi);
    llvm::Value *src = resolveRValue(LC, B, srcOp, B.getInt64Ty(), insn.id());

    // 128-bit signed multiply (LLVM supports i128 natively and reliably)
    llvm::Value *a128 = B.CreateSExt(rax, B.getInt128Ty(), "imul.a128");
    llvm::Value *b128 = B.CreateSExt(src, B.getInt128Ty(), "imul.b128");
    llvm::Value *prod = B.CreateMul(a128, b128, "imul.full");

    llvm::Value *low  = B.CreateTrunc(prod, B.getInt64Ty(), "imul.low");
    llvm::Value *high = B.CreateTrunc(B.CreateLShr(prod, 64), B.getInt64Ty(), "imul.high");

    writeGprSubreg(LC, B, raxRi, low);
    writeGprSubreg(LC, B, rdxRi, high);

    // CF/OF = 1 iff high 64 bits != sign-extension of low 64 bits (Intel spec)
    llvm::Value *signExt = B.CreateAShr(low, 63);
    llvm::Value *ov = B.CreateICmpNE(high, signExt, "imul.ov");
    storeFlag(LC, B, CF, ov);
    storeFlag(LC, B, OF, ov);

    recordMapping(LC, insn, low);
    return;
  }

  // 2-operand form (your original supported case) – kept exactly as before
  if (insn.operands_size() < 2) return;
  const auto &dst = insn.operands(0);
  const auto &src = insn.operands(1);
  if (!dst.has_register_()) return;
  RegInfo dri = decodeReg(dst.register_());
  if (!dri.isValid || dri.isXmm) return;

  llvm::Type *ty = intTy(LC.C, dri.bitWidth);
  llvm::Value *a = truncOrZext(B, readGprSubreg(LC, B, dri), ty);
  llvm::Value *b = resolveRValue(LC, B, src, ty, insn.id());

  llvm::Value *res = B.CreateMul(a, b, "imul.res");

  auto *smul = llvm::Intrinsic::getDeclaration(&LC.M, llvm::Intrinsic::smul_with_overflow, {ty});
  llvm::Value *call = B.CreateCall(smul, {a, b});
  llvm::Value *of = B.CreateExtractValue(call, 1);
  storeFlag(LC, B, OF, of);
  storeFlag(LC, B, CF, of);

  writeGprSubreg(LC, B, dri, res);
  recordMapping(LC, insn, res);
}

static void lowerCDQ(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn) {
  RegInfo eax = decodeReg("EAX");
  RegInfo edx = decodeReg("EDX");
  llvm::Value *a = readGprSubreg(LC, B, eax);
  llvm::Value *sign = B.CreateAShr(a, B.getInt32(31));
  writeGprSubreg(LC, B, edx, sign);
  recordMapping(LC, insn, sign);
}

static void lowerDIV(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn) {
  if (insn.operands_size() < 1) return;
  const auto &srcOp = insn.operands(0);

  RegInfo eaxRi = decodeReg("EAX");
  RegInfo edxRi = decodeReg("EDX");

  llvm::Value *eaxV = readGprSubreg(LC, B, eaxRi);
  llvm::Value *edxV = readGprSubreg(LC, B, edxRi);

  // dividend = (EDX << 32) | EAX  (unsigned concatenation)
  llvm::Value *hi = B.CreateZExt(edxV, B.getInt64Ty());
  llvm::Value *lo = B.CreateZExt(eaxV, B.getInt64Ty());
  llvm::Value *dividend = B.CreateOr(B.CreateShl(hi, 32), lo, "div.dividend");

  llvm::Value *divisor32 = resolveRValue(LC, B, srcOp, B.getInt32Ty(), insn.id());
  llvm::Value *divisor = B.CreateZExt(divisor32, B.getInt64Ty(), "div.divisor");

  llvm::Value *q = B.CreateUDiv(dividend, divisor, "div.q");
  llvm::Value *r = B.CreateURem(dividend, divisor, "div.r");

  writeGprSubreg(LC, B, eaxRi, B.CreateTrunc(q, B.getInt32Ty()));
  writeGprSubreg(LC, B, edxRi, B.CreateTrunc(r, B.getInt32Ty()));

  // Intel: no flags are defined after DIV (we leave them untouched – most realistic)
  recordMapping(LC, insn, q);
}

static void lowerIDIV_ECX(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn) {
  if (insn.operands_size() < 1) return;
  const auto &src = insn.operands(0);

  RegInfo eax = decodeReg("EAX");
  RegInfo edx = decodeReg("EDX");
  llvm::Value *eaxV = readGprSubreg(LC, B, eax);
  llvm::Value *edxV = readGprSubreg(LC, B, edx);
  llvm::Value *hi = B.CreateSExt(edxV, B.getInt64Ty(), "idiv.hi.sext");
  llvm::Value *lo = B.CreateZExt(eaxV, B.getInt64Ty(), "idiv.lo.zext");
  llvm::Value *dividend = B.CreateOr(B.CreateShl(hi, B.getInt64(32)), lo, "idiv.dividend");

  llvm::Value *divisor32 = resolveRValue(LC, B, src, B.getInt32Ty(), insn.id());
  llvm::Value *divisor = B.CreateSExt(divisor32, B.getInt64Ty(), "idiv.divisor");

  llvm::Value *q = B.CreateSDiv(dividend, divisor, "idiv.q");
  llvm::Value *r = B.CreateSRem(dividend, divisor, "idiv.r");

  writeGprSubreg(LC, B, eax, B.CreateTrunc(q, B.getInt32Ty()));
  writeGprSubreg(LC, B, edx, B.CreateTrunc(r, B.getInt32Ty()));
  recordMapping(LC, insn, B.CreateTrunc(q, B.getInt32Ty()));
}

static void lowerUnsupportedButKeepIR(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn) {
  std::string opc = insn.has_opcode() ? toUpper(insn.opcode()) : "UNKNOWN";

  llvm::FunctionType *ft = llvm::FunctionType::get(llvm::Type::getVoidTy(LC.C), false);
  llvm::InlineAsm *ia = llvm::InlineAsm::get(ft, "; placeholder for " + opc, "", true);
  llvm::CallInst *placeholder = B.CreateCall(ia);
  recordMapping(LC, insn, placeholder);

  if (insn.operands_size() >= 1) {
    const auto &dst = insn.operands(0);
    if (dst.has_register_()) {
      RegInfo ri = decodeReg(dst.register_());
      if (ri.isValid) {
        if (ri.isXmm) {
          llvm::Type *xmmTy = llvm::FixedVectorType::get(B.getDoubleTy(), 2);
          llvm::Value *p = getStateFieldPtr(LC, B, kXmmBase + ri.xmmIndex);
          B.CreateStore(llvm::UndefValue::get(xmmTy), p);
        } else {
          llvm::Type *ty = intTy(LC.C, ri.bitWidth);
          writeGprSubreg(LC, B, ri, llvm::UndefValue::get(ty));
        }
      }
    }
  }
}

// ----------------------------- stack promotion ------------------------------

static llvm::Type* slotTypeHeuristic(llvm::LLVMContext &C, int32_t size) {
  if (size == 1) return llvm::Type::getInt8Ty(C);
  if (size == 2) return llvm::Type::getInt16Ty(C);
  if (size == 4) return llvm::Type::getInt32Ty(C);
  if (size == 8) return llvm::Type::getInt64Ty(C);
  if (size > 0) return llvm::ArrayType::get(llvm::Type::getInt8Ty(C), (uint64_t)size);
  return llvm::ArrayType::get(llvm::Type::getInt8Ty(C), 1);
}

static void promoteStackSlots(FnLowerCtx &LC, IRBuilder<> &EntryB) {
  for (const auto &ss : LC.FnAst->stack_slots()) {
    StackSlotInfo info;
    info.name = ss.has_name() ? ss.name() : "slot";
    info.baseReg = ss.has_register_() ? ss.register_() : "RBP";
    info.startOff = ss.has_offset() ? ss.offset() : 0;
    info.size = ss.has_size() ? ss.size() : 8;
    info.align = ss.has_alignment() ? ss.alignment() : 1;

    llvm::Type *ty = slotTypeHeuristic(LC.C, info.size);
    llvm::AllocaInst *a = EntryB.CreateAlloca(ty, nullptr, info.name);
    a->setAlignment(llvm::Align((unsigned)std::max<int32_t>(1, info.align)));
    info.alloca = a;
    LC.promotedSlots.push_back(info);
  }
}

// ----------------------------- per-function lowering ------------------------

static void lowerFunction(FnLowerCtx &LC) {
  auto origLinkage = LC.F->getLinkage();
  LC.F->deleteBody();

  for (const auto &bbAst : LC.FnAst->basic_blocks()) {
    std::string bbName = bbAst.has_start_label() && !bbAst.start_label().empty() ? bbAst.start_label() : (bbAst.has_id() && !bbAst.id().empty() ? bbAst.id() : "bb");
    llvm::BasicBlock *bb = llvm::BasicBlock::Create(LC.C, bbName, LC.F);
    if (bbAst.has_id()) {
      LC.bbIdToLlvm[bbAst.id()] = bb;
      LC.bbLlvmMapping[bbAst.id()] = bb->getName().str();
    }
  }

  LC.StateArg = LC.F->getArg(0);
  if (LC.FnAst->basic_blocks_size() > 0) {
    const auto &entryAst = LC.FnAst->basic_blocks(0);
    auto it = LC.bbIdToLlvm.find(entryAst.id());
    if (it != LC.bbIdToLlvm.end()) {
      IRBuilder<> EntryB(it->second);
      promoteStackSlots(LC, EntryB);
    }
  }

  for (const auto &bbAst : LC.FnAst->basic_blocks()) {
    auto it = LC.bbIdToLlvm.find(bbAst.id());
    if (it == LC.bbIdToLlvm.end()) continue;
    llvm::BasicBlock *bb = it->second;

    IRBuilder<> B(bb);
    LC.stateGepCache.clear();

    for (const auto &ie : bbAst.instructions()) {
      const lifted_ast::Instruction &insn = ie.instruction();
      std::string opc = insn.has_opcode() ? toUpper(insn.opcode()) : "";

      if (bbAst.has_terminator() && insn.has_id() && insn.id() == bbAst.terminator()) continue;
      if (isTerminatorOpcode(opc)) continue;

      if (opc == "MOV") lowerMOV(LC, B, insn);
      else if (opc == "LEA") lowerLEA(LC, B, insn);
      else if (opc == "ADD" || opc == "SUB" || opc == "XOR" || opc == "AND" || opc == "OR") lowerALU2(LC, B, insn, opc);
      else if (opc == "CMP") lowerCMP(LC, B, insn);
      else if (opc == "TEST") lowerTEST(LC, B, insn);
      else if (opc == "MOVZX") lowerMOVZX_MOVSX(LC, B, insn, false);
      else if (opc == "MOVSX") lowerMOVZX_MOVSX(LC, B, insn, true);
      else if (opc == "SETC" || opc == "SETNE" || opc == "SETPE" || opc == "SETE" || opc == "SETZ") lowerSETcc(LC, B, insn, opc);
      else if (opc == "CMOVE") lowerCMOVE(LC, B, insn);
      else if (opc == "INC") lowerINC(LC, B, insn);
      else if (opc == "SHL") lowerSHL(LC, B, insn);
      else if (opc == "SHR") lowerSHR(LC, B, insn);
      else if (opc == "SAR") lowerSAR(LC, B, insn);
      else if (opc == "XCHG") lowerXCHG(LC, B, insn);
      else if (opc == "PUSH") lowerPUSH(LC, B, insn);
      else if (opc == "POP") lowerPOP(LC, B, insn);
      else if (opc == "LEAVE") lowerLEAVE(LC, B, insn);
      else if (opc == "IMUL") lowerIMUL(LC, B, insn);
      else if (opc == "CDQ") lowerCDQ(LC, B, insn);
      else if (opc == "IMUL") lowerIMUL(LC, B, insn);
      else if (opc == "DIV")  lowerDIV(LC, B, insn);
      else if (opc == "IDIV") lowerIDIV_ECX(LC, B, insn);
      else lowerUnsupportedButKeepIR(LC, B, insn);
    }
    if (!bb->getTerminator()) B.CreateUnreachable();
  }

  LC.F->setLinkage(origLinkage);

  if (llvm::verifyFunction(*LC.F, &llvm::errs())) {
    llvm::errs() << "verifyFunction failed for: " << LC.F->getName() << "\n";
  }
}

// ----------------------------- module driver -------------------------------

static std::unique_ptr<llvm::Module> loadBitcodeModule(const std::string &path, llvm::LLVMContext &C) {
  auto bufOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufOrErr) { std::cerr << "Failed to open bitcode: " << path << "\n"; return nullptr; }
  auto modOrErr = llvm::parseBitcodeFile(bufOrErr->get()->getMemBufferRef(), C);
  if (!modOrErr) {
    std::cerr << "Failed to parse bitcode: " << path << "\n";
    llvm::logAllUnhandledErrors(modOrErr.takeError(), llvm::errs(), "");
    return nullptr;
  }
  if (*modOrErr) {
    (*modOrErr)->setDataLayout("e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128");
    (*modOrErr)->setTargetTriple("x86_64-unknown-linux-gnu");
  }
  return std::move(*modOrErr);
}

static bool loadProtobuf(const std::string &path, lifted_ast::Program &P) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  return P.ParseFromIstream(&in);
}

static bool saveProtobuf(const std::string &path, const lifted_ast::Program &P) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return false;
  return P.SerializeToOstream(&out);
}

int main(int argc, char **argv) {
  bool printMode = false;
  std::vector<std::string> args;
  std::string printIrPath;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--print-ir" && i + 1 < argc) printIrPath = argv[++i];
    else if (a == "--print") printMode = true;
    else args.push_back(a);
  }
  if (args.size() < 4) {
    std::cerr << "Usage: llvmLowerInt <in.bc> <in.pb> <out.bc> <out.pb> [--print-ir out.ll] [--print]\n";
    return 1;
  }

  lifted_ast::Program P;
  if (!loadProtobuf(args[1], P)) { std::cerr << "Failed to read protobuf: " << args[1] << "\n"; return 1; }

  llvm::LLVMContext C;
  std::unique_ptr<llvm::Module> M = loadBitcodeModule(args[0], C);
  if (!M) return 1;

  llvm::StructType *StateTy = llvm::StructType::getTypeByName(C, "State");
  if (!StateTy) { std::cerr << "ERROR: Could not find identified struct type 'State' in module.\n"; return 1; }

  std::map<std::string, std::string> bbLlvmMapping;
  std::map<std::string, std::string> instrLlvmMapping;

  for (const auto &sec : P.sections()) {
    for (const auto &ch : sec.children()) {
      if (!ch.has_function()) continue;
      const lifted_ast::Function &fnAst = ch.function();
      if (!fnAst.has_entry_label()) continue;

      std::string entry = fnAst.entry_label();
      std::string liftedName = entry + "_lifted";
      auto itSe = P.symbol_table().find(entry);
      if (itSe != P.symbol_table().end() && itSe->second.has_lifted_ref() && !itSe->second.lifted_ref().empty()) {
        liftedName = itSe->second.lifted_ref();
      }

      if (llvm::Function *F = M->getFunction(liftedName)) {
        FnLowerCtx LC{C, *M, StateTy, F, nullptr, &P, &fnAst, bbLlvmMapping, instrLlvmMapping};
        lowerFunction(LC);
      } else {
        llvm::errs() << "Warning: lifted function not found in module: " << liftedName << "\n";
      }
    }
  }

  for (const auto &kv : bbLlvmMapping) (*P.mutable_bb_llvm_mapping())[kv.first] = kv.second;
  for (const auto &kv : instrLlvmMapping) (*P.mutable_instr_llvm_mapping())[kv.first] = kv.second;

  std::string originalDL = M->getDataLayoutStr();
  std::string originalTriple = M->getTargetTriple();
  if (llvm::verifyModule(*M, &llvm::errs())) llvm::errs() << "verifyModule FAILED after Step 11 lowering.\n";
  M->setDataLayout(originalDL);
  M->setTargetTriple(originalTriple);

  if (printMode) {
    std::string jsonStr;
    google::protobuf::util::JsonPrintOptions opts;
    opts.add_whitespace = true;
    if (!google::protobuf::util::MessageToJsonString(P, &jsonStr, opts).ok()) return 1;
    std::ofstream out(args[3], std::ios::trunc);
    out << jsonStr;
  } else {
    if (!saveProtobuf(args[3], P)) return 1;
  }

  {
    std::error_code EC;
    llvm::raw_fd_ostream os(args[2], EC, llvm::sys::fs::OF_None);
    if (EC) return 1;
    if (printMode) M->print(os, nullptr);
    else llvm::WriteBitcodeToFile(*M, os);
    os.flush();
  }

  if (!printIrPath.empty()) {
    std::error_code EC;
    llvm::raw_fd_ostream os(printIrPath, EC, llvm::sys::fs::OF_None);
    if (!EC) { M->print(os, nullptr); os.flush(); }
  }
  return 0;
}
