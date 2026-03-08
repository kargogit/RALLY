/*
llvmLowerFpAtomic.cpp (Implementation of Step 12)

Supported Instructions:

Floating-Point Data Movement
- MOVSS (XMM register ↔ memory)
- MOVSD (load from memory to XMM)

Floating-Point Arithmetic
- ADDSS
- ADDSD
- MULSS

Floating-Point Conversion
- CVTTSS2SI

Floating-Point Comparison / Flags
- UCOMISS
- COMISS

Atomic Operations
- XCHG (reg ↔ mem only)
- INC (with LOCK prefix on memory)
*/

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

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
#include <llvm/IR/Instructions.h>
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
#include <llvm/Support/raw_ostream.h>

#include "ast.pb.h"
#include <google/protobuf/struct.pb.h>
#include <google/protobuf/util/json_util.h>

using namespace llvm;

// ----------------------------- Utilities (from Step 11) -----------------------------

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
  auto isSpace = [](unsigned char c) { return std::isspace(c) != 0; };
  while (!s.empty() && isSpace((unsigned char)s.front())) s.erase(s.begin());
  while (!s.empty() && isSpace((unsigned char)s.back())) s.pop_back();
  if (s.empty()) return std::nullopt;

  int base = 10;
  if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) base = 16;

  char *end = nullptr;
  errno = 0;
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
    case V::kStringValue:
      return parseInt64Loose(v.string_value());
    default:
      return std::nullopt;
  }
}

// ----------------------------- %State layout -----------------------------

static constexpr unsigned kGprCount  = 16;
static constexpr unsigned kFlagCount = 9;
static constexpr unsigned kXmmCount  = 16;
static constexpr unsigned kFlagsBase = kGprCount;
static constexpr unsigned kXmmBase   = kGprCount + kFlagCount;

enum FlagIndex : unsigned {
  CF = 0, PF = 1, AF = 2, ZF = 3, SF = 4, OF = 5, DF = 6, IF_FLAG = 7, RF = 8,
};

// ----------------------------- Register decoding -----------------------------

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
  return (it == m.end()) ? -1 : it->second;
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

  if (r == "RAX") { setGpr("RAX",64,0); return ri; }
  if (r == "EAX") { setGpr("RAX",32,0); return ri; }
  if (r == "AX")  { setGpr("RAX",16,0); return ri; }
  if (r == "AL")  { setGpr("RAX",8,0); return ri; }
  if (r == "AH")  { setGpr("RAX",8,8); return ri; }

  if (r == "RBX") { setGpr("RBX",64,0); return ri; }
  if (r == "EBX") { setGpr("RBX",32,0); return ri; }
  if (r == "BX")  { setGpr("RBX",16,0); return ri; }
  if (r == "BL")  { setGpr("RBX",8,0); return ri; }
  if (r == "BH")  { setGpr("RBX",8,8); return ri; }

  if (r == "RCX") { setGpr("RCX",64,0); return ri; }
  if (r == "ECX") { setGpr("RCX",32,0); return ri; }
  if (r == "CX")  { setGpr("RCX",16,0); return ri; }
  if (r == "CL")  { setGpr("RCX",8,0); return ri; }
  if (r == "CH")  { setGpr("RCX",8,8); return ri; }

  if (r == "RDX") { setGpr("RDX",64,0); return ri; }
  if (r == "EDX") { setGpr("RDX",32,0); return ri; }
  if (r == "DX")  { setGpr("RDX",16,0); return ri; }
  if (r == "DL")  { setGpr("RDX",8,0); return ri; }
  if (r == "DH")  { setGpr("RDX",8,8); return ri; }

  if (r == "RSI") { setGpr("RSI",64,0); return ri; }
  if (r == "ESI") { setGpr("RSI",32,0); return ri; }
  if (r == "SI")  { setGpr("RSI",16,0); return ri; }
  if (r == "SIL") { setGpr("RSI",8,0); return ri; }

  if (r == "RDI") { setGpr("RDI",64,0); return ri; }
  if (r == "EDI") { setGpr("RDI",32,0); return ri; }
  if (r == "DI")  { setGpr("RDI",16,0); return ri; }
  if (r == "DIL") { setGpr("RDI",8,0); return ri; }

  if (r == "RBP") { setGpr("RBP",64,0); return ri; }
  if (r == "EBP") { setGpr("RBP",32,0); return ri; }
  if (r == "BP")  { setGpr("RBP",16,0); return ri; }
  if (r == "BPL") { setGpr("RBP",8,0); return ri; }

  if (r == "RSP") { setGpr("RSP",64,0); return ri; }
  if (r == "ESP") { setGpr("RSP",32,0); return ri; }
  if (r == "SP")  { setGpr("RSP",16,0); return ri; }
  if (r == "SPL") { setGpr("RSP",8,0); return ri; }

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
      if (suff.empty())       { setGpr(base,64,0); return ri; }
      if (suff == "D")        { setGpr(base,32,0); return ri; }
      if (suff == "W")        { setGpr(base,16,0); return ri; }
      if (suff == "B")        { setGpr(base,8,0);  return ri; }
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

// ----------------------------- Lowering context -----------------------------

struct FnLowerCtx {
  llvm::LLVMContext &C;
  llvm::Module &M;
  llvm::StructType *StateTy = nullptr;
  llvm::Function *F = nullptr;
  llvm::Value *StateArg = nullptr;
  const lifted_ast::Program *P = nullptr;
  const lifted_ast::Function *FnAst = nullptr;

  std::map<std::string, llvm::BasicBlock*> bbIdToLlvm;
  llvm::DenseMap<unsigned, llvm::Value*> stateGepCache;

  std::map<std::string, std::string> &instrLlvmMapping;

  FnLowerCtx(llvm::LLVMContext &c, llvm::Module &m, llvm::StructType *stateTy,
             llvm::Function *f, llvm::Value *stateArg,
             const lifted_ast::Program *p, const lifted_ast::Function *fnAst,
             std::map<std::string, std::string> &instrMap)
    : C(c), M(m), StateTy(stateTy), F(f), StateArg(stateArg),
      P(p), FnAst(fnAst), instrLlvmMapping(instrMap) {}
};

static llvm::Value* getStateFieldPtr(FnLowerCtx &LC, IRBuilder<> &B, unsigned fieldIdx) {
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
  return B.CreateBitCast(v, dstTy);
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
  llvm::Value *invMaskV = llvm::ConstantInt::get(i64, ~mask);

  llvm::Value *kept = B.CreateAnd(old, invMaskV, "merge.kept");
  llvm::Value *merged = B.CreateOr(kept, sub64, "merge");
  storeGpr64(LC, B, (unsigned)ri.gprIndex, merged);
}

// ----------------------------- Flag computation -----------------------------

static llvm::Value* computePF(IRBuilder<> &B, llvm::Value *resIntN) {
  llvm::Value *lo8 = B.CreateTrunc(resIntN, B.getInt8Ty(), "pf.lo8");
  llvm::Function *ctpop = llvm::Intrinsic::getDeclaration(
      B.GetInsertBlock()->getModule(), llvm::Intrinsic::ctpop, {B.getInt8Ty()});
  llvm::Value *pop = B.CreateCall(ctpop, {lo8}, "pf.pop");
  llvm::Value *lsb = B.CreateAnd(pop, B.getInt8(1), "pf.lsb");
  llvm::Value *odd = B.CreateICmpEQ(lsb, B.getInt8(1), "pf.odd");
  return B.CreateNot(odd, "pf");
}

// ----------------------------- Memory address resolution -----------------------------

struct MemAddr {
  llvm::Value *ptr = nullptr;
  llvm::Align align = llvm::Align(1);
  bool isSymbolic = false;
  std::string symName;
};

static llvm::Value* symbolAddressAsPtr(FnLowerCtx &LC, IRBuilder<> &B, const std::string &sym) {
  if (llvm::GlobalVariable *gv = LC.M.getNamedGlobal(sym)) {
    llvm::Type *vt = gv->getValueType();
    llvm::SmallVector<llvm::Value*, 2> idx{B.getInt32(0)};
    if (vt->isArrayTy() || vt->isStructTy() || vt->isVectorTy()) idx.push_back(B.getInt32(0));
    return B.CreateInBoundsGEP(vt, gv, idx, "sym.gep");
  }
  if (llvm::Function *fn = LC.M.getFunction(sym)) return fn;
  return llvm::UndefValue::get(llvm::PointerType::getUnqual(LC.C));
}

static MemAddr resolveMemAddress(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Operand &op, unsigned accessSize) {
  MemAddr out;
  out.ptr = llvm::UndefValue::get(llvm::PointerType::getUnqual(LC.C));
  out.align = llvm::Align(1);
  out.isSymbolic = false;

  if (!op.has_memory()) return out;
  const lifted_ast::Memory &m = op.memory();

  const std::string base = m.has_base() ? toUpper(m.base()) : "";
  const std::string index = m.has_index() ? toUpper(m.index()) : "";
  const int32_t scale = m.has_scale() ? m.scale() : 1;

  if (op.has_symbol_ref()) {
    llvm::Value *basePtr = symbolAddressAsPtr(LC, B, op.symbol_ref());
    out.isSymbolic = true;
    out.symName = op.symbol_ref();

    if (m.has_displacement()) {
      auto dispOpt = valueAsInt64(m.displacement());
      if (dispOpt && *dispOpt != 0) {
        basePtr = B.CreateGEP(B.getInt8Ty(), basePtr, B.getInt64(*dispOpt), "sym.disp");
      }
    }
    out.ptr = basePtr;
    out.align = llvm::Align(accessSize > 0 ? accessSize : 1);
    return out;
  }

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

// ----------------------------- Erasure & Metadata Utilities -----------------------------

static std::optional<std::string> getPlaceholderOpcode(llvm::CallInst *CI) {
  if (!CI) return std::nullopt;
  llvm::Value *callee = CI->getCalledOperand();
  auto *IA = llvm::dyn_cast<llvm::InlineAsm>(callee);
  if (!IA) return std::nullopt;

  std::string asmStr = IA->getAsmString();
  const char *prefix = "; placeholder for ";
  size_t pos = asmStr.find(prefix);
  if (pos == std::string::npos) return std::nullopt;

  std::string opc = asmStr.substr(pos + strlen(prefix));
  while (!opc.empty() && std::isspace((unsigned char)opc.back())) opc.pop_back();
  return opc;
}

static std::string getAstInstrId(llvm::Instruction *I) {
  if (!I) return "";
  llvm::MDNode *md = I->getMetadata("ast_instr_id");
  if (!md || md->getNumOperands() == 0) return "";
  if (auto *str = llvm::dyn_cast<llvm::MDString>(md->getOperand(0))) {
    return str->getString().str();
  }
  return "";
}

static std::vector<llvm::Instruction*> getAstIdRange(llvm::BasicBlock *BB, const std::string &instrId) {
  std::vector<llvm::Instruction*> range;
  if (instrId.empty()) return range;

  for (auto &I : *BB) {
    if (getAstInstrId(&I) == instrId) {
      range.push_back(&I);
    }
  }
  return range;
}

static void eraseOldCodeAndUsers(const std::vector<llvm::Instruction*> &roots,
                                  const std::string &instrId) {
  std::set<llvm::Instruction*> to_delete;
  std::vector<llvm::Instruction*> worklist = roots;

  while (!worklist.empty()) {
    llvm::Instruction *I = worklist.back();
    worklist.pop_back();

    if (!I) continue;
    if (!to_delete.insert(I).second) continue;

    for (llvm::User *U : I->users()) {
      if (auto *UI = llvm::dyn_cast<llvm::Instruction>(U)) {
        std::string uid = getAstInstrId(UI);
        if (uid.empty() || uid == instrId) {
          worklist.push_back(UI);
        }
      }
    }
  }

  std::vector<llvm::Instruction*> final_delete;
  for (auto *I : to_delete) {
    if (I->isTerminator()) continue;
    final_delete.push_back(I);
  }

  for (auto *I : final_delete) {
    if (!I->use_empty()) I->replaceAllUsesWith(llvm::UndefValue::get(I->getType()));
  }

  for (auto *I : final_delete) I->eraseFromParent();
}

static void attachPicRelocations(FnLowerCtx &LC, llvm::Instruction *I, const std::string &sym) {
  auto it = LC.P->symbol_table().find(sym);
  if (it != LC.P->symbol_table().end() && it->second.relocations_size() > 0) {
    bool isPic = false;
    for (const auto &rel : it->second.relocations()) {
      if (rel.pic()) { isPic = true; break; }
    }
    if (isPic) {
      I->setMetadata("pic_relocations", llvm::MDNode::get(LC.C, {llvm::MDString::get(LC.C, sym)}));
    }
  }
}

static void finalizeNewInstructions(FnLowerCtx &LC, llvm::BasicBlock *BB,
                                    llvm::Instruction *prev, llvm::Instruction *insertBefore,
                                    const std::string &instrId) {
  if (instrId.empty()) return;
  llvm::Instruction *start = prev ? prev->getNextNode() : &BB->front();
  if (!start || start == insertBefore) return;

  llvm::MDNode *md = llvm::MDNode::get(LC.C, {llvm::MDString::get(LC.C, instrId)});
  llvm::Instruction *primary = nullptr;

  for (llvm::Instruction *I = start; I != insertBefore; I = I->getNextNode()) {
    I->setMetadata("ast_instr_id", md);
    if (!primary && !I->getType()->isVoidTy()) {
      if (llvm::isa<llvm::AtomicRMWInst>(I) || llvm::isa<llvm::AtomicCmpXchgInst>(I) ||
          llvm::isa<llvm::FPMathOperator>(I) || llvm::isa<llvm::CallInst>(I) || llvm::isa<llvm::LoadInst>(I)) {
        primary = I;
      }
    }
  }

  if (!primary) {
    for (llvm::Instruction *I = start; I != insertBefore; I = I->getNextNode()) {
      if (!I->getType()->isVoidTy()) { primary = I; break; }
    }
  }

  if (primary && !primary->getType()->isVoidTy()) {
    std::string newName = "instr_" + instrId + "_lowered";
    primary->setName(newName);
    LC.instrLlvmMapping[instrId] = primary->getName().str();
  } else {
    LC.instrLlvmMapping.erase(instrId);
  }
}

// ----------------------------- Dummy store erasure -----------------------------

/// After erasing a placeholder call, scan forward from `start` and erase
/// orphaned dummy instructions: sequences of GEP (to %State) followed by
/// store of undef/zeroinitializer/constant-zero that lack a distinct
/// !ast_instr_id (i.e., they have no metadata or the same instrId as the
/// placeholder we just replaced). These are Step 11 leftovers that would
/// overwrite the values our real lowering just stored.
static void eraseDummyStoresAfter(llvm::Instruction *start,
                                  const std::string &placeholderInstrId) {
  if (!start) return;

  llvm::SmallVector<llvm::Instruction*, 8> toErase;
  llvm::Instruction *cur = start;

  while (cur && !cur->isTerminator()) {
    // Check if this is a store of a dummy value (undef or zero constant)
    if (auto *SI = llvm::dyn_cast<llvm::StoreInst>(cur)) {
      // Only erase if it has no ast_instr_id or the same one as the placeholder
      std::string sid = getAstInstrId(SI);
      if (!sid.empty() && sid != placeholderInstrId) break; // belongs to next real instruction

      llvm::Value *val = SI->getValueOperand();
      bool isDummy = false;

      if (llvm::isa<llvm::UndefValue>(val)) {
        isDummy = true;
      } else if (auto *CI = llvm::dyn_cast<llvm::ConstantInt>(val)) {
        isDummy = CI->isZero();
      } else if (auto *CAZ = llvm::dyn_cast<llvm::ConstantAggregateZero>(val)) {
        (void)CAZ;
        isDummy = true;
      }

      if (isDummy) {
        // Also check if the pointer operand is a GEP into %State that we
        // should clean up (only if it has no other users)
        llvm::Value *ptr = SI->getPointerOperand();
        toErase.push_back(cur);
        cur = cur->getNextNode();

        // Try to also erase the GEP if it's only used by the dummy store
        if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(ptr)) {
          if (GEP->hasOneUse()) { // the store we're about to erase is its only user
            toErase.push_back(GEP);
          }
        }
        continue;
      }
    }

    // Check if this is a GEP with no ast_instr_id (potential prelude to a dummy store)
    if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(cur)) {
      std::string gid = getAstInstrId(GEP);
      if (gid.empty() || gid == placeholderInstrId) {
        // Peek ahead to see if next instruction is a dummy store using this GEP
        llvm::Instruction *next = cur->getNextNode();
        if (next) {
          if (auto *SI = llvm::dyn_cast<llvm::StoreInst>(next)) {
            if (SI->getPointerOperand() == GEP) {
              llvm::Value *val = SI->getValueOperand();
              bool isDummy = llvm::isa<llvm::UndefValue>(val) ||
                             llvm::isa<llvm::ConstantAggregateZero>(val) ||
                             (llvm::isa<llvm::ConstantInt>(val) &&
                              llvm::cast<llvm::ConstantInt>(val)->isZero());
              std::string sid = getAstInstrId(SI);
              bool sameOrNoId = sid.empty() || sid == placeholderInstrId;

              if (isDummy && sameOrNoId) {
                toErase.push_back(SI);
                toErase.push_back(GEP);
                cur = next->getNextNode();
                continue;
              }
            }
          }
        }
        // GEP alone without a following dummy store - don't skip, stop scanning
        break;
      } else {
        break; // Different ast_instr_id - belongs to next real instruction
      }
    }

    // Any other instruction type - stop scanning
    break;
  }

  // Erase in reverse to handle use-before-def correctly (store before GEP)
  // But we need to be careful: erase stores first, then GEPs
  llvm::SmallVector<llvm::Instruction*, 4> stores, geps;
  for (auto *I : toErase) {
    if (llvm::isa<llvm::StoreInst>(I)) stores.push_back(I);
    else geps.push_back(I);
  }
  for (auto *I : stores) I->eraseFromParent();
  for (auto *I : geps) {
    if (I->use_empty()) I->eraseFromParent();
  }
}

// ----------------------------- FP Lowering Functions -----------------------------

static llvm::Value* loadXmm(FnLowerCtx &LC, IRBuilder<> &B, unsigned xmmIdx) {
  auto *xmmTy = llvm::FixedVectorType::get(B.getDoubleTy(), 2);
  llvm::Value *p = getStateFieldPtr(LC, B, kXmmBase + xmmIdx);
  return B.CreateLoad(xmmTy, p, "xmm.ld");
}

static void storeXmm(FnLowerCtx &LC, IRBuilder<> &B, unsigned xmmIdx, llvm::Value *vec) {
  llvm::Value *p = getStateFieldPtr(LC, B, kXmmBase + xmmIdx);
  B.CreateStore(vec, p);
}

static void lowerMOVSS_load(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn) {
  if (insn.operands_size() < 2) return;
  const auto &dst = insn.operands(0);
  const auto &src = insn.operands(1);

  RegInfo dri = decodeReg(dst.register_());
  MemAddr ma = resolveMemAddress(LC, B, src, 4);
  llvm::LoadInst *ld = B.CreateLoad(B.getFloatTy(), ma.ptr, "movss.ld");
  if (ma.isSymbolic) attachPicRelocations(LC, ld, ma.symName);

  auto *f4Ty = llvm::FixedVectorType::get(B.getFloatTy(), 4);
  llvm::Value *f4Vec = B.CreateInsertElement(llvm::ConstantAggregateZero::get(f4Ty), ld, (uint64_t)0, "movss.ins");
  llvm::Value *d2Vec = B.CreateBitCast(f4Vec, llvm::FixedVectorType::get(B.getDoubleTy(), 2), "movss.bc");
  storeXmm(LC, B, dri.xmmIndex, d2Vec);
}

static void lowerMOVSS_store(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn) {
  if (insn.operands_size() < 2) return;
  const auto &dst = insn.operands(0);
  const auto &src = insn.operands(1);

  RegInfo sri = decodeReg(src.register_());
  llvm::Value *d2Vec = loadXmm(LC, B, sri.xmmIndex);
  llvm::Value *f4Vec = B.CreateBitCast(d2Vec, llvm::FixedVectorType::get(B.getFloatTy(), 4), "movss.bc");
  llvm::Value *f32Val = B.CreateExtractElement(f4Vec, (uint64_t)0, "movss.ext");

  MemAddr ma = resolveMemAddress(LC, B, dst, 4);
  llvm::StoreInst *st = B.CreateStore(f32Val, ma.ptr);
  if (ma.isSymbolic) attachPicRelocations(LC, st, ma.symName);
}

static void lowerMULSS(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn) {
  if (insn.operands_size() < 2) return;
  RegInfo dri = decodeReg(insn.operands(0).register_());
  auto *f4Ty = llvm::FixedVectorType::get(B.getFloatTy(), 4);
  llvm::Value *f4VecDst = B.CreateBitCast(loadXmm(LC, B, dri.xmmIndex), f4Ty, "mulss.dst.bc");
  llvm::Value *f0Dst = B.CreateExtractElement(f4VecDst, (uint64_t)0, "mulss.dst.f0");
  llvm::Value *f0Src = nullptr;
  if (insn.operands(1).has_register_()) {
    llvm::Value *f4VecSrc = B.CreateBitCast(loadXmm(LC, B, decodeReg(insn.operands(1).register_()).xmmIndex), f4Ty);
    f0Src = B.CreateExtractElement(f4VecSrc, (uint64_t)0, "mulss.src.f0");
  } else if (insn.operands(1).has_memory()) {
    MemAddr ma = resolveMemAddress(LC, B, insn.operands(1), 4);
    llvm::LoadInst *ld = B.CreateLoad(B.getFloatTy(), ma.ptr, "mulss.src.ld");
    if (ma.isSymbolic) attachPicRelocations(LC, ld, ma.symName);
    f0Src = ld;
  }
  llvm::Value *prod = B.CreateFMul(f0Dst, f0Src ? f0Src : f0Dst, "mulss.prod");
  llvm::Value *f4VecNew = B.CreateInsertElement(f4VecDst, prod, (uint64_t)0, "mulss.ins");
  storeXmm(LC, B, dri.xmmIndex, B.CreateBitCast(f4VecNew, llvm::FixedVectorType::get(B.getDoubleTy(), 2)));
}

static void lowerADDSS(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn) {
  if (insn.operands_size() < 2) return;
  RegInfo dri = decodeReg(insn.operands(0).register_());

  auto *f4Ty = llvm::FixedVectorType::get(B.getFloatTy(), 4);
  llvm::Value *f4VecDst = B.CreateBitCast(loadXmm(LC, B, dri.xmmIndex), f4Ty, "addss.dst.bc");
  llvm::Value *f0Dst = B.CreateExtractElement(f4VecDst, (uint64_t)0, "addss.dst.f0");
  llvm::Value *f0Src = nullptr;

  if (insn.operands(1).has_register_()) {
    llvm::Value *f4VecSrc = B.CreateBitCast(loadXmm(LC, B, decodeReg(insn.operands(1).register_()).xmmIndex), f4Ty);
    f0Src = B.CreateExtractElement(f4VecSrc, (uint64_t)0, "addss.src.f0");
  } else if (insn.operands(1).has_memory()) {
    MemAddr ma = resolveMemAddress(LC, B, insn.operands(1), 4);
    llvm::LoadInst *ld = B.CreateLoad(B.getFloatTy(), ma.ptr, "addss.src.ld");
    if (ma.isSymbolic) attachPicRelocations(LC, ld, ma.symName);
    f0Src = ld;
  }

  llvm::Value *sum = B.CreateFAdd(f0Dst, f0Src ? f0Src : f0Dst, "addss.sum");
  llvm::Value *f4VecNew = B.CreateInsertElement(f4VecDst, sum, (uint64_t)0, "addss.ins");
  storeXmm(LC, B, dri.xmmIndex, B.CreateBitCast(f4VecNew, llvm::FixedVectorType::get(B.getDoubleTy(), 2)));
}

static void lowerCVTTSS2SI(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn) {
  if (insn.operands_size() < 2) return;
  RegInfo dri = decodeReg(insn.operands(0).register_());
  llvm::Value *f32Val = nullptr;

  if (insn.operands(1).has_register_()) {
    llvm::Value *f4Vec = B.CreateBitCast(loadXmm(LC, B, decodeReg(insn.operands(1).register_()).xmmIndex), llvm::FixedVectorType::get(B.getFloatTy(), 4));
    f32Val = B.CreateExtractElement(f4Vec, (uint64_t)0, "cvttss.f0");
  } else if (insn.operands(1).has_memory()) {
    MemAddr ma = resolveMemAddress(LC, B, insn.operands(1), 4);
    llvm::LoadInst *ld = B.CreateLoad(B.getFloatTy(), ma.ptr, "cvttss.ld");
    if (ma.isSymbolic) attachPicRelocations(LC, ld, ma.symName);
    f32Val = ld;
  }

  if (f32Val) writeGprSubreg(LC, B, dri, B.CreateFPToSI(f32Val, intTy(LC.C, dri.bitWidth), "cvttss.tosi"));
}

static void lowerMOVSD_load(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn) {
  if (insn.operands_size() < 2) return;
  RegInfo dri = decodeReg(insn.operands(0).register_());

  MemAddr ma = resolveMemAddress(LC, B, insn.operands(1), 8);
  llvm::LoadInst *ld = B.CreateLoad(B.getDoubleTy(), ma.ptr, "movsd.ld");
  if (ma.isSymbolic) attachPicRelocations(LC, ld, ma.symName);

  storeXmm(LC, B, dri.xmmIndex, B.CreateInsertElement(loadXmm(LC, B, dri.xmmIndex), ld, (uint64_t)0, "movsd.ins"));
}

static void lowerADDSD(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn) {
  if (insn.operands_size() < 2) return;
  RegInfo dri = decodeReg(insn.operands(0).register_());

  llvm::Value *d2VecDst = loadXmm(LC, B, dri.xmmIndex);
  llvm::Value *f0Dst = B.CreateExtractElement(d2VecDst, (uint64_t)0, "addsd.dst.f0");
  llvm::Value *f0Src = nullptr;

  if (insn.operands(1).has_register_()) {
    f0Src = B.CreateExtractElement(loadXmm(LC, B, decodeReg(insn.operands(1).register_()).xmmIndex), (uint64_t)0, "addsd.src.f0");
  } else if (insn.operands(1).has_memory()) {
    MemAddr ma = resolveMemAddress(LC, B, insn.operands(1), 8);
    llvm::LoadInst *ld = B.CreateLoad(B.getDoubleTy(), ma.ptr, "addsd.src.ld");
    if (ma.isSymbolic) attachPicRelocations(LC, ld, ma.symName);
    f0Src = ld;
  }

  llvm::Value *sum = B.CreateFAdd(f0Dst, f0Src ? f0Src : f0Dst, "addsd.sum");
  storeXmm(LC, B, dri.xmmIndex, B.CreateInsertElement(d2VecDst, sum, (uint64_t)0, "addsd.ins"));
}

static void lowerUCOMISS(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn, bool ordered) {
  if (insn.operands_size() < 2) return;
  auto *f4Ty = llvm::FixedVectorType::get(B.getFloatTy(), 4);
  llvm::Value *f0a = nullptr, *f0b = nullptr;

  if (insn.operands(0).has_register_()) {
    f0a = B.CreateExtractElement(B.CreateBitCast(loadXmm(LC, B, decodeReg(insn.operands(0).register_()).xmmIndex), f4Ty), (uint64_t)0);
  }

  if (insn.operands(1).has_register_()) {
    f0b = B.CreateExtractElement(B.CreateBitCast(loadXmm(LC, B, decodeReg(insn.operands(1).register_()).xmmIndex), f4Ty), (uint64_t)0);
  } else if (insn.operands(1).has_memory()) {
    MemAddr ma = resolveMemAddress(LC, B, insn.operands(1), 4);
    llvm::LoadInst *ld = B.CreateLoad(B.getFloatTy(), ma.ptr, "ucomiss.ld");
    if (ma.isSymbolic) attachPicRelocations(LC, ld, ma.symName);
    f0b = ld;
  }

  if (!f0a || !f0b) return;

  llvm::Value *unord = B.CreateFCmpUNO(f0a, f0b, "ucomiss.unord");
  storeFlag(LC, B, CF, B.CreateOr(unord, B.CreateFCmpOLT(f0a, f0b, "ucomiss.lt"), "ucomiss.cf"));
  storeFlag(LC, B, ZF, B.CreateOr(unord, B.CreateFCmpOEQ(f0a, f0b, "ucomiss.eq"), "ucomiss.zf"));
  storeFlag(LC, B, PF, unord);
  storeFlag(LC, B, OF, B.getFalse());
  storeFlag(LC, B, SF, B.getFalse());
  storeFlag(LC, B, AF, B.getFalse());
}

// ----------------------------- Atomic Lowering Functions -----------------------------

static bool isXchgWithMemory(const lifted_ast::Instruction &insn) {
  std::string opc = insn.has_opcode() ? toUpper(insn.opcode()) : "";
  if (opc != "XCHG") return false;
  for (int i = 0; i < insn.operands_size(); ++i) {
    if (insn.operands(i).has_memory()) return true;
  }
  return false;
}

static bool hasLockPrefix(const lifted_ast::Instruction &insn) {
  return insn.has_prefix() && toUpper(insn.prefix()) == "LOCK";
}

static void lowerAtomicXCHG(FnLowerCtx &LC, const lifted_ast::Instruction &insn, llvm::BasicBlock *BB) {
  if (insn.operands_size() < 2) return;
  const lifted_ast::Operand *memOp = nullptr, *regOp = nullptr;

  if (insn.operands(0).has_memory() && insn.operands(1).has_register_()) {
    memOp = &insn.operands(0); regOp = &insn.operands(1);
  } else if (insn.operands(1).has_memory() && insn.operands(0).has_register_()) {
    memOp = &insn.operands(1); regOp = &insn.operands(0);
  }
  if (!memOp || !regOp) return;

  RegInfo rri = decodeReg(regOp->register_());
  std::string instrId = insn.has_id() ? insn.id() : "";
  std::vector<llvm::Instruction*> oldInstrs = getAstIdRange(BB, instrId);
  llvm::Instruction *insertBefore = oldInstrs.empty() ? nullptr : oldInstrs.front();
  if (!insertBefore) return;

  llvm::Instruction *prev = insertBefore->getPrevNode();
  IRBuilder<> B(insertBefore);
  LC.stateGepCache.clear();

  unsigned accessSize = rri.bitWidth / 8;
  MemAddr ma = resolveMemAddress(LC, B, *memOp, accessSize);

  llvm::AtomicRMWInst *rmw = B.CreateAtomicRMW(
      llvm::AtomicRMWInst::Xchg, ma.ptr, truncOrZext(B, readGprSubreg(LC, B, rri), intTy(LC.C, rri.bitWidth)),
      llvm::MaybeAlign(accessSize), llvm::AtomicOrdering::SequentiallyConsistent);
  if (ma.isSymbolic) attachPicRelocations(LC, rmw, ma.symName);

  writeGprSubreg(LC, B, rri, rmw);

  finalizeNewInstructions(LC, BB, prev, insertBefore, instrId);
  eraseOldCodeAndUsers(oldInstrs, instrId);
}

static void lowerAtomicINC(FnLowerCtx &LC, const lifted_ast::Instruction &insn, llvm::BasicBlock *BB) {
  if (insn.operands_size() < 1 || !insn.operands(0).has_memory()) return;
  std::string instrId = insn.has_id() ? insn.id() : "";
  std::vector<llvm::Instruction*> oldInstrs = getAstIdRange(BB, instrId);
  llvm::Instruction *insertBefore = oldInstrs.empty() ? nullptr : oldInstrs.front();
  if (!insertBefore) return;

  llvm::Instruction *prev = insertBefore->getPrevNode();
  IRBuilder<> B(insertBefore);
  LC.stateGepCache.clear();

  unsigned accessSize = std::max((unsigned)1, memSizeBytesFromOperand(insn.operands(0)));
  MemAddr ma = resolveMemAddress(LC, B, insn.operands(0), accessSize);
  llvm::Type *opTy = intTy(LC.C, accessSize * 8);
  llvm::Value *one = llvm::ConstantInt::get(opTy, 1);

  llvm::AtomicRMWInst *oldVal = B.CreateAtomicRMW(
      llvm::AtomicRMWInst::Add, ma.ptr, one,
      llvm::MaybeAlign(accessSize), llvm::AtomicOrdering::SequentiallyConsistent);
  if (ma.isSymbolic) attachPicRelocations(LC, oldVal, ma.symName);

  llvm::Value *newVal = B.CreateAdd(oldVal, one, "atomic.inc.new");
  storeFlag(LC, B, ZF, B.CreateICmpEQ(newVal, llvm::ConstantInt::get(opTy, 0), "zf"));
  storeFlag(LC, B, SF, B.CreateICmpSLT(newVal, llvm::ConstantInt::get(opTy, 0), "sf"));
  storeFlag(LC, B, PF, computePF(B, newVal));
  storeFlag(LC, B, OF, B.CreateExtractValue(B.CreateCall(llvm::Intrinsic::getDeclaration(&LC.M, llvm::Intrinsic::sadd_with_overflow, {opTy}), {oldVal, one}), 1, "of"));
  storeFlag(LC, B, AF, B.CreateICmpNE(B.CreateAnd(B.CreateXor(B.CreateXor(oldVal, one), newVal), llvm::ConstantInt::get(opTy, 0x10)), llvm::ConstantInt::get(opTy, 0)));

  // INC does not modify CF - preserve previous value
  llvm::Value *cfPtr = getStateFieldPtr(LC, B, kFlagsBase + (unsigned)CF);
  llvm::Value *prevCf = B.CreateLoad(B.getInt1Ty(), cfPtr, "prev.cf");
  B.CreateStore(prevCf, cfPtr);

  finalizeNewInstructions(LC, BB, prev, insertBefore, instrId);
  eraseOldCodeAndUsers(oldInstrs, instrId);
}

// ----------------------------- Main Processing -----------------------------

static void processFunction(FnLowerCtx &LC) {
  if (!LC.F || LC.F->empty()) return;

  std::map<std::string, const lifted_ast::Instruction*> idToAstInsn;
  for (const auto &bbAst : LC.FnAst->basic_blocks()) {
    for (const auto &ie : bbAst.instructions()) {
      if (ie.instruction().has_id()) idToAstInsn[ie.instruction().id()] = &ie.instruction();
    }
  }

  for (auto &BB : *LC.F) LC.bbIdToLlvm[BB.getName().str()] = &BB;
  LC.StateArg = LC.F->getArg(0);

  // Pass 1: Handle structured atomic instructions
  for (const auto &bbAst : LC.FnAst->basic_blocks()) {
    std::string bbName = bbAst.has_start_label() && !bbAst.start_label().empty() ? bbAst.start_label() : (bbAst.has_id() && !bbAst.id().empty() ? bbAst.id() : "");
    llvm::BasicBlock *BB = nullptr;
    if (auto it = LC.bbIdToLlvm.find(bbName); it != LC.bbIdToLlvm.end()) BB = it->second;
    if (!BB) continue;

    for (const auto &ie : bbAst.instructions()) {
      const lifted_ast::Instruction &insn = ie.instruction();
      std::string opc = insn.has_opcode() ? toUpper(insn.opcode()) : "";

      if (isXchgWithMemory(insn)) { lowerAtomicXCHG(LC, insn, BB); continue; }
      if (hasLockPrefix(insn) && opc == "INC") lowerAtomicINC(LC, insn, BB);
    }
  }

  // Pass 2: Find and replace FP placeholders
  for (auto &BB : *LC.F) {
    std::vector<llvm::CallInst*> placeholders;
    for (auto &I : BB) {
      if (auto *CI = llvm::dyn_cast<llvm::CallInst>(&I)) {
        if (getPlaceholderOpcode(CI)) placeholders.push_back(CI);
      }
    }

    for (auto *CI : placeholders) {
      auto opcOpt = getPlaceholderOpcode(CI);
      if (!opcOpt) continue;

      std::string opc = toUpper(*opcOpt);
      std::string instrId = getAstInstrId(CI);

      auto astIt = idToAstInsn.find(instrId);
      if (astIt == idToAstInsn.end()) continue;

      std::vector<llvm::Instruction*> oldInstrs = getAstIdRange(&BB, instrId);
      llvm::Instruction *insertBefore = oldInstrs.empty() ? CI : oldInstrs.front();
      llvm::Instruction *prev = insertBefore->getPrevNode();

      // Record the instruction immediately after the last old instruction
      // so we can scan for dummy stores after erasure
      llvm::Instruction *afterOldCode = nullptr;
      if (!oldInstrs.empty()) {
        afterOldCode = oldInstrs.back()->getNextNode();
      } else {
        afterOldCode = CI->getNextNode();
      }

      IRBuilder<> B(insertBefore);
      LC.stateGepCache.clear();

      bool handled = false;
      const lifted_ast::Instruction &insn = *astIt->second;

      if (opc == "MOVSS" && insn.operands_size() >= 2) {
        if (insn.operands(0).has_register_() && decodeReg(insn.operands(0).register_()).isXmm && insn.operands(1).has_memory()) { lowerMOVSS_load(LC, B, insn); handled = true; }
        else if (insn.operands(0).has_memory() && insn.operands(1).has_register_() && decodeReg(insn.operands(1).register_()).isXmm) { lowerMOVSS_store(LC, B, insn); handled = true; }
      } else if (opc == "ADDSS") { lowerADDSS(LC, B, insn); handled = true; }
      else if (opc == "MULSS") { lowerMULSS(LC, B, insn); handled = true; }
      else if (opc == "CVTTSS2SI") { lowerCVTTSS2SI(LC, B, insn); handled = true; }
      else if (opc == "MOVSD" && insn.operands_size() >= 2 && insn.operands(0).has_register_() && decodeReg(insn.operands(0).register_()).isXmm && insn.operands(1).has_memory()) { lowerMOVSD_load(LC, B, insn); handled = true; }
      else if (opc == "ADDSD") { lowerADDSD(LC, B, insn); handled = true; }
      else if (opc == "UCOMISS" || opc == "COMISS") { lowerUCOMISS(LC, B, insn, opc == "COMISS"); handled = true; }

      if (handled) {
        finalizeNewInstructions(LC, &BB, prev, insertBefore, instrId);
        eraseOldCodeAndUsers(oldInstrs, instrId);

        // Erase orphaned dummy stores that Step 11 emitted after the placeholder.
        // These lack !ast_instr_id and store undef/zero, overwriting our real values.
        eraseDummyStoresAfter(afterOldCode, instrId);
      }
    }
  }
}

// ----------------------------- Module driver -----------------------------

static std::unique_ptr<llvm::Module> loadBitcodeModule(const std::string &path, llvm::LLVMContext &C) {
  auto bufOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufOrErr) return nullptr;
  auto modOrErr = llvm::parseBitcodeFile(bufOrErr->get()->getMemBufferRef(), C);
  if (!modOrErr) {
    llvm::logAllUnhandledErrors(modOrErr.takeError(), llvm::errs(), "");
    return nullptr;
  }
  return std::move(*modOrErr);
}

static bool loadProtobuf(const std::string &path, lifted_ast::Program &P) {
  std::ifstream in(path, std::ios::binary);
  return in && P.ParseFromIstream(&in);
}

static bool saveProtobuf(const std::string &path, const lifted_ast::Program &P) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  return out && P.SerializeToOstream(&out);
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

  if (args.size() < 4) return 1;

  lifted_ast::Program P;
  if (!loadProtobuf(args[1], P)) return 1;

  llvm::LLVMContext C;
  std::unique_ptr<llvm::Module> M = loadBitcodeModule(args[0], C);
  if (!M) return 1;

  llvm::StructType *StateTy = llvm::StructType::getTypeByName(C, "State");
  if (!StateTy) return 1;

  std::map<std::string, std::string> instrLlvmMapping;
  for (const auto &kv : P.instr_llvm_mapping()) instrLlvmMapping[kv.first] = kv.second;

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
        FnLowerCtx LC{C, *M, StateTy, F, nullptr, &P, &fnAst, instrLlvmMapping};
        processFunction(LC);
      }
    }
  }

  for (const auto &kv : instrLlvmMapping) (*P.mutable_instr_llvm_mapping())[kv.first] = kv.second;
  llvm::verifyModule(*M, &llvm::errs());

  if (printMode) {
    std::string jsonStr;
    google::protobuf::util::JsonPrintOptions opts; opts.add_whitespace = true;
    if (google::protobuf::util::MessageToJsonString(P, &jsonStr, opts).ok()) {
      std::ofstream(args[3], std::ios::trunc) << jsonStr;
    }
  } else {
    saveProtobuf(args[3], P);
  }

  {
    std::error_code EC;
    llvm::raw_fd_ostream os(args[2], EC, llvm::sys::fs::OF_None);
    if (!EC) { printMode ? M->print(os, nullptr) : llvm::WriteBitcodeToFile(*M, os); os.flush(); }
  }

  if (!printIrPath.empty()) {
    std::error_code EC;
    llvm::raw_fd_ostream os(printIrPath, EC, llvm::sys::fs::OF_None);
    if (!EC) { M->print(os, nullptr); os.flush(); }
  }

  return 0;
}
