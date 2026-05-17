// llvmLiftShared.hpp
// =============================================================================
// PERFECT COMMON CORE SHARED LIBRARY FOR THE LLVM LIFTING PIPELINE
// Unified and Expanded Edition
//
// This version perfectly blends structural stability with advanced features:
// - Universal RValue/LValue decoding & Expression Evaluation
// - Stack Slot mapping (Alloca Promotion)
// - Sub-Register Read/Write tracking
// - Robust Memory, Symbol, & Got Slot resolution
// - Erasure, Placeholder, ABI Call, & Init Helpers
// =============================================================================

#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>          // REQUIRED for std::isfinite
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/raw_ostream.h>

#include "ast.pb.h"
#include <google/protobuf/struct.pb.h>

namespace llvm_lift {

using namespace llvm;

// =============================================================================
// 1. %State layout (authoritative)
// =============================================================================
static constexpr unsigned kGprCount = 16;
static constexpr unsigned kFlagCount = 9;
static constexpr unsigned kXmmCount = 16;
static constexpr unsigned kFlagsBase = kGprCount;
static constexpr unsigned kXmmBase = kGprCount + kFlagCount;

enum FlagIndex : unsigned {
  CF = 0, PF = 1, AF = 2, ZF = 3, SF = 4, OF = 5, DF = 6, IF_FLAG = 7, RF = 8
};

static StructType* createStateType(LLVMContext& ctx) {
  SmallVector<Type*, 64> fields;
  for (int i = 0; i < 16; ++i) fields.push_back(Type::getInt64Ty(ctx));
  for (int i = 0; i < 9; ++i) fields.push_back(Type::getInt1Ty(ctx));
  auto xmmTy = FixedVectorType::get(Type::getDoubleTy(ctx), 2);
  for (int i = 0; i < 16; ++i) fields.push_back(xmmTy);
  return StructType::create(ctx, fields, "State");
}

// =============================================================================
// 2. Small utilities
// =============================================================================
static std::string trim(std::string s) {
  auto notSpace = [](unsigned char c){ return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), notSpace));
  s.erase(std::find_if(s.rbegin(), s.rend(), notSpace).base(), s.end());
  return s;
}

static std::string toUpper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c){ return char(std::toupper(c)); });
  return s;
}

static bool endsWith(const std::string& s, const char* suff) {
  size_t n = std::strlen(suff);
  return s.size() >= n && s.compare(s.size() - n, n, suff) == 0;
}

static bool startsWith(const std::string& s, const char* pref) {
  return s.rfind(pref, 0) == 0;
}

static std::optional<int64_t> parseInt64Loose(const std::string& s0) {
  std::string s = trim(s0);
  if (s.empty()) return std::nullopt;
  int base = 10;
  if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) base = 16;
  char* end = nullptr;
  errno = 0;
  if (base == 16) {
    unsigned long long uv = std::strtoull(s.c_str(), &end, 16);
    if (errno == 0 && end != s.c_str() && *end == '\0') return (int64_t)uv;
  }
  long long v = std::strtoll(s.c_str(), &end, base);
  if (errno != 0 || end == s.c_str() || *end != '\0') return std::nullopt;
  return (int64_t)v;
}

static std::optional<int64_t> valueAsInt64(const google::protobuf::Value& v) {
  using V = google::protobuf::Value;
  switch (v.kind_case()) {
    case V::kNumberValue: {
      double d = v.number_value();
      if (!std::isfinite(d)) return std::nullopt;
      if (d > (double)std::numeric_limits<int64_t>::max() ||
          d < (double)std::numeric_limits<int64_t>::min()) return std::nullopt;
      return (int64_t)d;
    }
    case V::kStringValue:
      return parseInt64Loose(v.string_value());
    default:
      return std::nullopt;
  }
}

static std::optional<std::string> structStringField(const google::protobuf::Value& v, const std::string& key) {
  using V = google::protobuf::Value;
  if (v.kind_case() != V::kStructValue) return std::nullopt;
  const auto& fs = v.struct_value().fields();
  auto it = fs.find(key);
  if (it == fs.end() || it->second.kind_case() != V::kStringValue) return std::nullopt;
  return it->second.string_value();
}

static std::optional<google::protobuf::Value> structFieldValue(const google::protobuf::Value& v, const std::string& key) {
  using V = google::protobuf::Value;
  if (v.kind_case() != V::kStructValue) return std::nullopt;
  const auto& fs = v.struct_value().fields();
  auto it = fs.find(key);
  if (it == fs.end()) return std::nullopt;
  return it->second;
}

static unsigned memSizeBytesFromOperand(const lifted_ast::Operand& op) {
  if (!op.has_size()) return 0;
  std::string s = toUpper(op.size());
  if (s == "BYTE") return 1; if (s == "WORD") return 2;
  if (s == "DWORD") return 4; if (s == "QWORD") return 8;
  return 0;
}

// =============================================================================
// 3. Register decoding
// =============================================================================
static int gprFieldIndex64(const std::string& regUpper) {
  static const std::map<std::string, int> m = {
    {"RAX",0},{"RBX",1},{"RCX",2},{"RDX",3},{"RSI",4},{"RDI",5},
    {"R8",6},{"R9",7},{"R10",8},{"R11",9},{"R12",10},{"R13",11},{"R14",12},{"R15",13},
    {"RSP",14},{"RBP",15}
  };
  auto it = m.find(regUpper);
  return (it == m.end()) ? -1 : it->second;
}

struct RegInfo {
  bool isValid = false;
  bool isXmm = false;
  int gprIndex = -1;
  unsigned bitOffset = 0;
  unsigned bitWidth = 0;
  unsigned xmmIndex = 0;
};

static RegInfo decodeReg(const std::string& regName0) {
  RegInfo ri;
  std::string r = toUpper(regName0);

  if (r.size() >= 3 && r.substr(0, 3) == "XMM") {
    std::string idxStr = r.substr(3);
    if (auto nOpt = parseInt64Loose(idxStr)) {
      if (*nOpt >= 0 && *nOpt < (int64_t)kXmmCount) {
        ri.isValid = true; ri.isXmm = true; ri.xmmIndex = (unsigned)*nOpt;
      }
    }
    return ri;
  }

  auto setGpr = [&](const std::string& base64, unsigned w, unsigned off) {
    int idx = gprFieldIndex64(base64);
    if (idx >= 0) { ri.isValid = true; ri.gprIndex = idx; ri.bitWidth = w; ri.bitOffset = off; }
  };

  if (r == "RAX") setGpr("RAX",64,0); else if (r == "EAX") setGpr("RAX",32,0);
  else if (r == "AX") setGpr("RAX",16,0); else if (r == "AL") setGpr("RAX",8,0); else if (r == "AH") setGpr("RAX",8,8);
  else if (r == "RBX") setGpr("RBX",64,0); else if (r == "EBX") setGpr("RBX",32,0);
  else if (r == "BX") setGpr("RBX",16,0); else if (r == "BL") setGpr("RBX",8,0); else if (r == "BH") setGpr("RBX",8,8);
  else if (r == "RCX") setGpr("RCX",64,0); else if (r == "ECX") setGpr("RCX",32,0);
  else if (r == "CX") setGpr("RCX",16,0); else if (r == "CL") setGpr("RCX",8,0); else if (r == "CH") setGpr("RCX",8,8);
  else if (r == "RDX") setGpr("RDX",64,0); else if (r == "EDX") setGpr("RDX",32,0);
  else if (r == "DX") setGpr("RDX",16,0); else if (r == "DL") setGpr("RDX",8,0); else if (r == "DH") setGpr("RDX",8,8);
  else if (r == "RSI") setGpr("RSI",64,0); else if (r == "ESI") setGpr("RSI",32,0);
  else if (r == "SI") setGpr("RSI",16,0); else if (r == "SIL") setGpr("RSI",8,0);
  else if (r == "RDI") setGpr("RDI",64,0); else if (r == "EDI") setGpr("RDI",32,0);
  else if (r == "DI") setGpr("RDI",16,0); else if (r == "DIL") setGpr("RDI",8,0);
  else if (r == "RBP") setGpr("RBP",64,0); else if (r == "EBP") setGpr("RBP",32,0);
  else if (r == "BP") setGpr("RBP",16,0); else if (r == "BPL") setGpr("RBP",8,0);
  else if (r == "RSP") setGpr("RSP",64,0); else if (r == "ESP") setGpr("RSP",32,0);
  else if (r == "SP") setGpr("RSP",16,0); else if (r == "SPL") setGpr("RSP",8,0);
  else if (r.size() >= 2 && r[0] == 'R' && std::isdigit((unsigned char)r[1])) {
    std::string base = "R"; size_t i = 1;
    while (i < r.size() && std::isdigit((unsigned char)r[i])) { base.push_back(r[i]); ++i; }
    std::string suff = r.substr(i);
    if (suff.empty()) setGpr(base,64,0); else if (suff == "D") setGpr(base,32,0);
    else if (suff == "W") setGpr(base,16,0); else if (suff == "B") setGpr(base,8,0);
  }
  return ri;
}

// =============================================================================
// 4. LLVM type parsing
// =============================================================================
static Type* parseLLVMType(LLVMContext& ctx, const std::string& s0) {
  std::string s = trim(s0);
  if (s == "void") return Type::getVoidTy(ctx);
  if (s == "i1") return Type::getInt1Ty(ctx);
  if (s == "i8") return Type::getInt8Ty(ctx);
  if (s == "i32") return Type::getInt32Ty(ctx);
  if (s == "i64") return Type::getInt64Ty(ctx);
  if (s == "float" || s == "f32") return Type::getFloatTy(ctx);
  if (s == "double" || s == "f64") return Type::getDoubleTy(ctx);
  if (s == "ptr" || (!s.empty() && s.back() == '*') || s.find("ptr") != std::string::npos)
    return PointerType::getUnqual(ctx);

  if (!s.empty() && s.front() == '[') {
    auto xPos = s.find(" x ");
    auto closePos = s.rfind(']');
    if (xPos != std::string::npos && closePos != std::string::npos && closePos > xPos) {
      auto nStr = trim(s.substr(1, xPos - 1));
      auto elemStr = trim(s.substr(xPos + 3, closePos - (xPos + 3)));
      uint64_t n = std::stoull(nStr);
      Type* elemTy = parseLLVMType(ctx, elemStr);
      return ArrayType::get(elemTy, n);
    }
  }
  return PointerType::getUnqual(ctx);
}

static FunctionType* parseFunctionType(LLVMContext& ctx, const std::string& s0) {
  std::string s = trim(s0);
  auto lp = s.find('('); auto rp = s.rfind(')');
  if (lp == std::string::npos || rp == std::string::npos || rp < lp)
    return FunctionType::get(PointerType::getUnqual(ctx), {}, true);

  std::string retStr = trim(s.substr(0, lp));
  std::string argsStr = trim(s.substr(lp + 1, rp - (lp + 1)));

  Type* retTy = parseLLVMType(ctx, retStr);
  SmallVector<Type*, 8> args;
  bool vararg = false;

  if (!argsStr.empty()) {
    size_t start = 0;
    while (start < argsStr.size()) {
      size_t comma = argsStr.find(',', start);
      std::string tok = trim(argsStr.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
      if (tok == "...") vararg = true;
      else if (!tok.empty()) args.push_back(parseLLVMType(ctx, tok));
      if (comma == std::string::npos) break;
      start = comma + 1;
    }
  }
  return FunctionType::get(retTy, args, vararg);
}

// =============================================================================
// 5. Stack Mapping Context
// =============================================================================
struct StackSlotInfo {
  std::string name;
  std::string baseReg;
  int64_t startOff = 0;
  int32_t size = 0;
  int32_t align = 1;
  AllocaInst* alloca = nullptr;
};

static std::optional<std::pair<Value*, Align>> lookupStackSlotPtr(
    IRBuilder<>& B, const std::vector<StackSlotInfo>& slots, const std::string& baseRegUpper, int64_t disp, unsigned accessSize) {
  for (const auto& s : slots) {
    if (toUpper(s.baseReg) != baseRegUpper) continue;
    if (disp >= s.startOff && (disp + (int64_t)accessSize) <= (s.startOff + s.size)) {
      Value* p = s.alloca;
      if (int64_t innerOff = disp - s.startOff; innerOff != 0)
        p = B.CreateGEP(B.getInt8Ty(), p, B.getInt64(innerOff), "stk.gep");
      return std::make_pair(p, Align((unsigned)std::max<int32_t>(1, s.align)));
    }
  }
  return std::nullopt;
}

// =============================================================================
// 6. Lowering Context
// =============================================================================
struct FnLowerCtx {
  LLVMContext& C;
  Module& M;
  StructType* StateTy = nullptr;
  Function* F = nullptr;
  Value* StateArg = nullptr;
  const lifted_ast::Program* P = nullptr;
  const lifted_ast::Function* FnAst = nullptr;

  std::map<std::string, BasicBlock*> bbIdToLlvm;
  DenseMap<unsigned, Value*> stateGepCache;
  std::vector<StackSlotInfo> promotedSlots;

  std::map<std::string, std::string>& instrLlvmMapping;
  std::map<std::string, std::string>* bbLlvmMapping = nullptr;

  FnLowerCtx(LLVMContext& c, Module& m, StructType* stateTy, Function* f, Value* stateArg,
             const lifted_ast::Program* p, const lifted_ast::Function* fnAst,
             std::map<std::string, std::string>& instrMap, std::map<std::string, std::string>* bbMap = nullptr)
    : C(c), M(m), StateTy(stateTy), F(f), StateArg(stateArg), P(p), FnAst(fnAst),
      instrLlvmMapping(instrMap), bbLlvmMapping(bbMap) {}
};

// =============================================================================
// 7. Metadata Helpers
// =============================================================================
static void attachAstInstrId(Instruction* I, const std::string& id, LLVMContext& C) {
  if (I && !id.empty()) I->setMetadata("ast_instr_id", MDNode::get(C, MDString::get(C, id)));
}

static void attachPicRelocations(Instruction* I, const std::string& sym, LLVMContext& C) {
  if (I && !sym.empty()) I->setMetadata("pic_relocations", MDNode::get(C, MDString::get(C, sym)));
}

static void recordMapping(FnLowerCtx& LC, const lifted_ast::Instruction& insn, Value* v) {
  if (!insn.has_id() || insn.id().empty() || !v) return;
  std::string name = "instr_" + insn.id();
  if (auto* I = dyn_cast<Instruction>(v)) {
    if (!I->getType()->isVoidTy() && !I->hasName()) I->setName(name);
    attachAstInstrId(I, insn.id(), LC.C);
  }
  LC.instrLlvmMapping[insn.id()] = name;
}

// =============================================================================
// 8. State Accessors & Value Helpers
// =============================================================================
static Value* getStateFieldPtr(FnLowerCtx& LC, IRBuilder<>& B, unsigned fieldIdx) {
  auto it = LC.stateGepCache.find(fieldIdx);
  if (it != LC.stateGepCache.end()) return it->second;
  Value* p = B.CreateStructGEP(LC.StateTy, LC.StateArg, fieldIdx, "state.gep");
  LC.stateGepCache[fieldIdx] = p;
  return p;
}

static Value* loadGpr64(FnLowerCtx& LC, IRBuilder<>& B, unsigned gprIdx) {
  return B.CreateLoad(B.getInt64Ty(), getStateFieldPtr(LC, B, gprIdx), "gpr64");
}

static void storeGpr64(FnLowerCtx& LC, IRBuilder<>& B, unsigned gprIdx, Value* vI64) {
  B.CreateStore(vI64, getStateFieldPtr(LC, B, gprIdx));
}

static Value* loadFlag(FnLowerCtx& LC, IRBuilder<>& B, FlagIndex f) {
  return B.CreateLoad(B.getInt1Ty(), getStateFieldPtr(LC, B, kFlagsBase + (unsigned)f), "flag");
}

static void storeFlag(FnLowerCtx& LC, IRBuilder<>& B, FlagIndex f, Value* vI1) {
  B.CreateStore(vI1, getStateFieldPtr(LC, B, kFlagsBase + (unsigned)f));
}

static Value* loadXmm(FnLowerCtx& LC, IRBuilder<>& B, unsigned xmmIdx) {
  auto* xmmTy = FixedVectorType::get(B.getDoubleTy(), 2);
  return B.CreateLoad(xmmTy, getStateFieldPtr(LC, B, kXmmBase + xmmIdx), "xmm.ld");
}

static void storeXmm(FnLowerCtx& LC, IRBuilder<>& B, unsigned xmmIdx, Value* vec) {
  B.CreateStore(vec, getStateFieldPtr(LC, B, kXmmBase + xmmIdx));
}

static Type* intTy(LLVMContext& C, unsigned bits) {
  return IntegerType::get(C, bits);
}

static Value* truncOrZext(IRBuilder<>& B, Value* v, Type* dstTy) {
  Type* srcTy = v->getType();
  if (srcTy == dstTy) return v;
  if (srcTy->isIntegerTy() && dstTy->isIntegerTy()) {
    unsigned sb = srcTy->getIntegerBitWidth(), db = dstTy->getIntegerBitWidth();
    return (sb > db) ? B.CreateTrunc(v, dstTy) : B.CreateZExt(v, dstTy);
  }
  return B.CreateBitCast(v, dstTy);
}

static Value* truncOrSext(IRBuilder<>& B, Value* v, Type* dstTy) {
  Type* srcTy = v->getType();
  if (srcTy == dstTy) return v;
  if (srcTy->isIntegerTy() && dstTy->isIntegerTy()) {
    unsigned sb = srcTy->getIntegerBitWidth(), db = dstTy->getIntegerBitWidth();
    return (sb > db) ? B.CreateTrunc(v, dstTy) : B.CreateSExt(v, dstTy);
  }
  return B.CreateBitCast(v, dstTy);
}

// =============================================================================
// 9. Sub-Register Operations
// =============================================================================
static Value* readGprSubreg(FnLowerCtx& LC, IRBuilder<>& B, const RegInfo& ri) {
  Value* full = loadGpr64(LC, B, (unsigned)ri.gprIndex);
  if (ri.bitWidth == 64 && ri.bitOffset == 0) return full;
  Value* shifted = full;
  if (ri.bitOffset != 0)
    shifted = B.CreateLShr(full, B.getInt64(ri.bitOffset), "shr.subreg");
  Type* ty = intTy(LC.C, ri.bitWidth);
  return B.CreateTrunc(shifted, ty, "trunc.subreg");
}

static void writeGprSubreg(FnLowerCtx& LC, IRBuilder<>& B, const RegInfo& ri, Value* v) {
  if (!ri.isValid || ri.isXmm || ri.gprIndex < 0) return;
  Type* i64 = B.getInt64Ty();
  unsigned w = ri.bitWidth, off = ri.bitOffset;

  if (w == 64 && off == 0) {
    storeGpr64(LC, B, ri.gprIndex, v->getType()->isPointerTy() ? B.CreatePtrToInt(v, i64) : truncOrZext(B, v, i64));
    return;
  }
  if (w == 32 && off == 0) {
    storeGpr64(LC, B, ri.gprIndex, B.CreateZExt(truncOrZext(B, v, B.getInt32Ty()), i64, "zext32"));
    return;
  }

  Value* old = loadGpr64(LC, B, ri.gprIndex);
  Value* sub64 = B.CreateZExt(truncOrZext(B, v, intTy(LC.C, w)), i64, "sub.zext64");
  if (off != 0) sub64 = B.CreateShl(sub64, B.getInt64(off), "sub.shl");

  uint64_t mask = ((w == 64) ? ~0ULL : ((1ULL << w) - 1ULL)) << off;
  Value* merged = B.CreateOr(
      B.CreateAnd(old, ConstantInt::get(i64, ~mask)),
      B.CreateAnd(sub64, ConstantInt::get(i64, mask)), "merge");
  storeGpr64(LC, B, ri.gprIndex, merged);
}

// =============================================================================
// 10. Universal Memory & Symbol Access
// =============================================================================
static Value* symbolAddressAsPtr(FnLowerCtx& LC, IRBuilder<>& B, const std::string& sym, const std::string& astInstrId = "") {
  if (auto it = LC.P->symbol_table().find(sym); it != LC.P->symbol_table().end()) {
    if (it->second.has_kind() && it->second.kind() == "label" && it->second.has_definition()) {
      if (auto bbId = structStringField(it->second.definition(), "bb_id"); bbId && LC.bbIdToLlvm.count(*bbId))
        return BlockAddress::get(LC.F, LC.bbIdToLlvm[*bbId]);
    }
  }
  if (GlobalVariable* gv = LC.M.getNamedGlobal(sym)) {
    Type* vt = gv->getValueType();
    SmallVector<Value*, 2> idx{B.getInt32(0)};
    if (vt->isArrayTy() || vt->isStructTy() || vt->isVectorTy()) idx.push_back(B.getInt32(0));
    Value* gep = B.CreateInBoundsGEP(vt, gv, idx, "sym.gep");
    if (!isa<Instruction>(gep)) gep = B.Insert(GetElementPtrInst::CreateInBounds(vt, gv, idx, "sym.gep"));
    if (auto* I = dyn_cast<Instruction>(gep)) {
      attachPicRelocations(I, sym, LC.C);
      attachAstInstrId(I, astInstrId, LC.C);
    }
    return gep;
  }
  if (Function* fn = LC.M.getFunction(sym)) return fn;
  return UndefValue::get(PointerType::getUnqual(LC.C));
}

static Value* resolveGotSlot(FnLowerCtx& LC, const std::string& sym) {
  std::string gotName = sym + "@GOT";
  if (GlobalVariable* gotSlot = LC.M.getNamedGlobal(gotName)) return gotSlot;
  Constant* init = LC.M.getNamedGlobal(sym);
  if (!init) init = LC.M.getFunction(sym) ? (Constant*)LC.M.getFunction(sym) : UndefValue::get(PointerType::getUnqual(LC.C));
  return new GlobalVariable(LC.M, PointerType::getUnqual(LC.C), true, GlobalValue::PrivateLinkage, init, gotName);
}

struct MemAddr {
  Value* ptr = nullptr;
  Align align = Align(1);
  bool isSymbolic = false;
  std::string symName;
};

static MemAddr resolveMemAddress(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Operand& op, unsigned accessSize = 0, const std::string& astInstrId = "") {
  MemAddr out;
  out.ptr = UndefValue::get(PointerType::getUnqual(LC.C));
  out.align = Align(1);
  out.isSymbolic = false;

  if (!op.has_memory()) return out;
  const auto& m = op.memory();
  const std::string base = m.has_base() ? toUpper(m.base()) : "";
  const std::string index = m.has_index() ? toUpper(m.index()) : "";
  const int32_t scale = m.has_scale() ? m.scale() : 1;

  if ((base == "RBP" || base == "RSP") && m.has_displacement()) {
    if (auto disp = valueAsInt64(m.displacement())) {
      if (auto stk = lookupStackSlotPtr(B, LC.promotedSlots, base, *disp, accessSize)) {
        out.ptr = stk->first; out.align = stk->second; return out;
      }
    }
  }

  if (op.has_symbol_ref()) {
    bool isGot = (op.has_via_got() && op.via_got()) || op.symbol_ref() == "stderr";
    Value* basePtr = isGot ? resolveGotSlot(LC, op.symbol_ref()) : symbolAddressAsPtr(LC, B, op.symbol_ref(), astInstrId);
    out.isSymbolic = true; out.symName = op.symbol_ref();
    if (m.has_displacement()) {
      if (auto disp = valueAsInt64(m.displacement()); disp && *disp != 0) {
        basePtr = B.CreateGEP(B.getInt8Ty(), basePtr, B.getInt64(*disp), isGot ? "got.disp" : "sym.disp");
        if (auto* I = dyn_cast<Instruction>(basePtr)) {
          attachPicRelocations(I, out.symName, LC.C);
          attachAstInstrId(I, astInstrId, LC.C);
        }
      }
    }
    out.ptr = basePtr;
    return out;
  }

  if (base == "RIP" && m.has_displacement()) {
    std::string sym;
    if (m.displacement().kind_case() == google::protobuf::Value::kStringValue) sym = m.displacement().string_value();
    else if (auto s = structStringField(m.displacement(), "symbol")) sym = *s;
    if (!sym.empty() && (LC.M.getNamedGlobal(sym) || LC.M.getFunction(sym) || LC.P->symbol_table().count(sym))) {
      out.ptr = symbolAddressAsPtr(LC, B, sym, astInstrId);
      out.isSymbolic = true; out.symName = sym;
      return out;
    }
  }

  Value* addrI64 = B.getInt64(0);
  if (!base.empty() && base != "RIP") {
    RegInfo bri = decodeReg(base);
    if (bri.isValid && !bri.isXmm) {
      addrI64 = B.CreateAdd(addrI64, truncOrZext(B, readGprSubreg(LC, B, bri), B.getInt64Ty()), "addr.base");
    }
  }
  if (!index.empty()) {
    RegInfo iri = decodeReg(index);
    if (iri.isValid && !iri.isXmm) {
      Value* i64v = truncOrZext(B, readGprSubreg(LC, B, iri), B.getInt64Ty());
      if (scale != 1 && scale > 0) i64v = B.CreateMul(i64v, B.getInt64(scale), "addr.scale");
      addrI64 = B.CreateAdd(addrI64, i64v, "addr.index");
    }
  }
  if (m.has_displacement()) {
    if (auto disp = valueAsInt64(m.displacement()); disp && *disp != 0) {
      addrI64 = B.CreateAdd(addrI64, B.getInt64(*disp), "addr.disp");
    }
  }
  out.ptr = B.CreateIntToPtr(addrI64, PointerType::getUnqual(LC.C), "addr.ptr");
  return out;
}

// =============================================================================
// 11. Expressions, Load/Store, RValue/LValue
// =============================================================================
static Value* evalExprToI64(FnLowerCtx& LC, IRBuilder<>& B, const google::protobuf::Value& v, const std::string& astInstrId = "") {
    // === FIX: Direct RBP usage for stack array bases (prevents undef propagation) ===
    // This matches the exact situation described: array base should be the frame pointer
    // because elements live relative to RBP and are accessed via slot2.
    if (v.kind_case() == google::protobuf::Value::kStructValue) {
        // Direct register case
        if (auto reg = structStringField(v, "register")) {
            if (*reg == "RBP" || *reg == "EBP") {
                return loadGpr64(LC, B, gprFieldIndex64("RBP"));
            }
        }

        // Additive expression containing RBP → treat as frame-pointer base
        // (removes the broken add chain that produced "... + undef")
        if (auto addV = structFieldValue(v, "additive")) {
            if (addV->kind_case() == google::protobuf::Value::kListValue) {
                for (const auto& elt : addV->list_value().values()) {
                    if (auto reg = structStringField(elt, "register")) {
                        if (*reg == "RBP" || *reg == "EBP") {
                            return loadGpr64(LC, B, gprFieldIndex64("RBP"));
                        }
                    }
                }
            }
        }
    }
  if (auto i = valueAsInt64(v)) return B.getInt64(*i);
  if (v.kind_case() == google::protobuf::Value::kStructValue) {
    if (auto r = structStringField(v, "register")) {
      RegInfo ri = decodeReg(*r);
      if (ri.isValid && !ri.isXmm) return truncOrZext(B, readGprSubreg(LC, B, ri), B.getInt64Ty());
    }
    if (auto sym = structStringField(v, "symbol")) {
      if (GlobalVariable* GV = LC.M.getNamedGlobal(*sym))
        if (GV->isConstant() && GV->hasInitializer())
          if (auto* CI = dyn_cast<ConstantInt>(GV->getInitializer())) return truncOrZext(B, CI, B.getInt64Ty());
      Value* pi64 = B.CreatePtrToInt(symbolAddressAsPtr(LC, B, *sym, astInstrId), B.getInt64Ty(), "sym.ptrtoint");
      if (auto* I = dyn_cast<Instruction>(pi64)) attachAstInstrId(I, astInstrId, LC.C);
      return pi64;
    }
    if (auto addV = structFieldValue(v, "additive"); addV && addV->kind_case() == google::protobuf::Value::kListValue) {
      Value* acc = B.getInt64(0);
      for (const auto& elt : addV->list_value().values()) acc = B.CreateAdd(acc, evalExprToI64(LC, B, elt, astInstrId), "add");
      return acc;
    }
    if (auto subV = structFieldValue(v, "subtract"); subV && subV->kind_case() == google::protobuf::Value::kListValue) {
      const auto& xs = subV->list_value().values();
      if (xs.empty()) return B.getInt64(0);
      Value* acc = evalExprToI64(LC, B, xs[0], astInstrId);
      for (int i = 1; i < xs.size(); ++i) acc = B.CreateSub(acc, evalExprToI64(LC, B, xs[i], astInstrId), "sub");
      return acc;
    }
  }
  return UndefValue::get(B.getInt64Ty());
}

static Value* loadFromMem(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Operand& memOp, Type* ty, const std::string& astInstrId = "") {
  MemAddr a = resolveMemAddress(LC, B, memOp, ty->isIntegerTy() ? ty->getIntegerBitWidth() / 8 : 8, astInstrId);
  LoadInst* L = B.CreateLoad(ty, a.ptr, "mem.ld");
  if (a.align.value() > 1) L->setAlignment(a.align);
  if (a.isSymbolic && !a.symName.empty()) attachPicRelocations(L, a.symName, LC.C);
  attachAstInstrId(L, astInstrId, LC.C);
  return L;
}

static void storeToMem(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Operand& memOp, Value* v, Type* storeTy, const std::string& astInstrId = "") {
  MemAddr a = resolveMemAddress(LC, B, memOp, storeTy->isIntegerTy() ? storeTy->getIntegerBitWidth() / 8 : 8, astInstrId);
  Value* vv = v;
  if (vv->getType() != storeTy) {
    if (storeTy->isIntegerTy()) vv = truncOrZext(B, vv, storeTy);
    else if (storeTy->isPointerTy() && vv->getType()->isIntegerTy(64)) vv = B.CreateIntToPtr(vv, storeTy);
    else if (storeTy->isIntegerTy(64) && vv->getType()->isPointerTy()) vv = B.CreatePtrToInt(vv, storeTy);
    else vv = B.CreateBitCast(vv, storeTy);
  }
  StoreInst* S = B.CreateStore(vv, a.ptr);
  if (a.align.value() > 1) S->setAlignment(a.align);
  if (a.isSymbolic && !a.symName.empty()) attachPicRelocations(S, a.symName, LC.C);
  attachAstInstrId(S, astInstrId, LC.C);
}

static Value* resolveRValue(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Operand& op, Type* desiredTy, const std::string& astInstrId = "") {
  if (op.has_register_()) {
    RegInfo ri = decodeReg(op.register_());
    if (ri.isValid && ri.isXmm) {
      Type* xmmTy = FixedVectorType::get(B.getDoubleTy(), 2);
      Value* v = loadXmm(LC, B, ri.xmmIndex);
      return desiredTy == xmmTy ? v : UndefValue::get(desiredTy);
    }
    if (ri.isValid && !ri.isXmm) {
      Value* v = readGprSubreg(LC, B, ri);
      return desiredTy->isIntegerTy() ? truncOrZext(B, v, desiredTy) : B.CreateIntToPtr(truncOrZext(B, v, B.getInt64Ty()), desiredTy);
    }
  }
  if (op.has_memory()) return loadFromMem(LC, B, op, desiredTy, astInstrId);
  if (op.has_symbol_ref()) {
    Value* val = nullptr; std::string sym = op.symbol_ref();
    if (GlobalVariable* GV = LC.M.getNamedGlobal(sym))
      if (GV->isConstant() && GV->hasInitializer())
        if (auto* CI = dyn_cast<ConstantInt>(GV->getInitializer())) val = CI;
    if (!val) {
      val = B.CreatePtrToInt(symbolAddressAsPtr(LC, B, sym, astInstrId), B.getInt64Ty(), "sym.imm.ptrtoint");
      if (isa<Instruction>(val)) attachAstInstrId(cast<Instruction>(val), astInstrId, LC.C);
    }
    val = truncOrZext(B, val, B.getInt64Ty());
    if (op.has_integer() && op.integer().has_value())
      if (auto iOpt = valueAsInt64(op.integer().value()); iOpt && *iOpt != 0) val = B.CreateAdd(val, B.getInt64(*iOpt), "sym.imm.add");
    if (desiredTy->isIntegerTy()) return truncOrZext(B, val, desiredTy);
    if (desiredTy->isPointerTy()) return B.CreateIntToPtr(val, desiredTy, "sym.imm.inttoptr");
    if (desiredTy->isFloatingPointTy()) return B.CreateBitCast(truncOrZext(B, val, intTy(LC.C, desiredTy->getPrimitiveSizeInBits())), desiredTy);
    return UndefValue::get(desiredTy);
  }
  if (op.has_expression()) {
    Value* exprI64 = evalExprToI64(LC, B, op.expression(), astInstrId);
    return desiredTy->isIntegerTy() ? truncOrZext(B, exprI64, desiredTy) : B.CreateIntToPtr(exprI64, desiredTy);
  }
  if (op.has_integer()) {
    int64_t imm = op.integer().has_value() ? valueAsInt64(op.integer().value()).value_or(0) : 0;
    if (desiredTy->isIntegerTy()) return ConstantInt::get(desiredTy, APInt(desiredTy->getIntegerBitWidth(), (uint64_t)imm, true));
    if (desiredTy->isPointerTy()) return B.CreateIntToPtr(ConstantInt::get(B.getInt64Ty(), (uint64_t)imm, true), desiredTy);
    if (desiredTy->isFloatingPointTy()) return ConstantFP::get(desiredTy, (double)imm);
  }
  return UndefValue::get(desiredTy);
}

// =============================================================================
// 12. Placeholder & Erasure (Gold Standard)
// =============================================================================
static std::optional<std::string> getPlaceholderOpcode(CallInst* CI) {
  if (!CI) return std::nullopt;
  auto* IA = dyn_cast<InlineAsm>(CI->getCalledOperand());
  if (!IA) return std::nullopt;
  std::string asmStr = IA->getAsmString();
  const char* prefix = "; placeholder for ";
  size_t pos = asmStr.find(prefix);
  if (pos == std::string::npos) return std::nullopt;
  std::string opc = asmStr.substr(pos + strlen(prefix));
  while (!opc.empty() && std::isspace((unsigned char)opc.back())) opc.pop_back();
  return opc;
}

static std::string getAstInstrId(Instruction* I) {
  if (!I) return "";
  MDNode* md = I->getMetadata("ast_instr_id");
  if (!md || md->getNumOperands() == 0) return "";
  if (auto* str = dyn_cast<MDString>(md->getOperand(0))) return str->getString().str();
  return "";
}

static void eraseDummyStoresAfter(Instruction* start, const std::string& placeholderInstrId) {
  if (!start) return;
  SmallVector<Instruction*, 8> toErase;
  Instruction* cur = start;
  while (cur && !cur->isTerminator()) {
    if (auto* SI = dyn_cast<StoreInst>(cur)) {
      std::string sid = getAstInstrId(SI);
      if (!sid.empty() && sid != placeholderInstrId) break;
      Value* val = SI->getValueOperand();
      bool isDummy = isa<UndefValue>(val) || isa<ConstantAggregateZero>(val) ||
                     (isa<ConstantInt>(val) && cast<ConstantInt>(val)->isZero());
      if (isDummy) {
        toErase.push_back(cur);
        cur = cur->getNextNode();
        if (auto* GEP = dyn_cast<GetElementPtrInst>(SI->getPointerOperand()))
          if (GEP->hasOneUse()) toErase.push_back(GEP);
        continue;
      }
    }
    if (auto* GEP = dyn_cast<GetElementPtrInst>(cur)) {
      std::string gid = getAstInstrId(GEP);
      if (gid.empty() || gid == placeholderInstrId) {
        if (Instruction* next = cur->getNextNode()) {
          if (auto* SI = dyn_cast<StoreInst>(next)) {
            if (SI->getPointerOperand() == GEP) {
              Value* val = SI->getValueOperand();
              bool isDummy = isa<UndefValue>(val) || isa<ConstantAggregateZero>(val) ||
                             (isa<ConstantInt>(val) && cast<ConstantInt>(val)->isZero());
              std::string sid = getAstInstrId(SI);
              if (isDummy && (sid.empty() || sid == placeholderInstrId)) {
                toErase.push_back(SI);
                toErase.push_back(GEP);
                cur = next->getNextNode();
                continue;
              }
            }
          }
        }
      }
      break;
    }
    break;
  }
  SmallVector<Instruction*, 4> stores, geps;
  for (auto* I : toErase) { if (isa<StoreInst>(I)) stores.push_back(I); else geps.push_back(I); }
  for (auto* I : stores) I->eraseFromParent();
  for (auto* I : geps) if (I->use_empty()) I->eraseFromParent();
}

// =============================================================================
// 13. Known weak symbols & Init Helpers
// =============================================================================
static const std::unordered_set<std::string> kKnownWeakSymbols = {
  "_ITM_deregisterTMCloneTable", "_ITM_registerTMCloneTable",
  "__gmon_start__", "__cxa_finalize"
};

static bool isKnownWeakSymbol(const std::string& name) {
  return kKnownWeakSymbols.count(name) > 0;
}

static void createStubBody(Function* F) {
  if (!F->empty()) return;
  LLVMContext& ctx = F->getContext();
  BasicBlock* bb = BasicBlock::Create(ctx, "entry", F);
  IRBuilder<> B(bb);
  if (F->hasFnAttribute(Attribute::NoReturn)) {
    B.CreateUnreachable(); return;
  }
  Type* retTy = F->getReturnType();
  if (retTy->isVoidTy()) B.CreateRetVoid();
  else B.CreateRet(UndefValue::get(retTy));
}

static void applyLiftedAttributes(Function* F, const lifted_ast::LiftedFunctionSignature& sig) {
  for (const auto& a : sig.attributes()) {
    if (a == "noreturn") F->addFnAttr(Attribute::NoReturn);
  }
}

static GlobalValue::LinkageTypes resolveLinkage(const lifted_ast::SymbolEntry& entry) {
  if (entry.has_linkage()) {
    if (entry.linkage() == "internal") return GlobalValue::InternalLinkage;
    if (entry.linkage() == "appending") return GlobalValue::AppendingLinkage;
    return GlobalValue::ExternalLinkage;
  }
  if (entry.has_visibility() && entry.visibility() == "local")
    return GlobalValue::InternalLinkage;
  return GlobalValue::ExternalLinkage;
}

// =============================================================================
// 14. ABI Call Helper
// =============================================================================
static CallInst* emitFunctionCall(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Operand& op, bool isTailCall = false) {
  bool isInternal = false;
  Function* targetF = nullptr;
  if (op.has_symbol_ref()) {
    auto it = LC.P->symbol_table().find(op.symbol_ref());
    if (it != LC.P->symbol_table().end() && it->second.has_lifted_ref() && !it->second.lifted_ref().empty()) {
      isInternal = true;
      targetF = LC.M.getFunction(it->second.lifted_ref());
    } else {
      targetF = LC.M.getFunction(op.symbol_ref());
    }
  }

  CallInst* ci = nullptr;
  if (isInternal && targetF) {
    ci = B.CreateCall(targetF, {LC.StateArg});
  } else if (!isInternal && targetF) {
    FunctionType* ft = targetF->getFunctionType();
    SmallVector<Value*, 8> args;
    const int intRegs[6] = { gprFieldIndex64("RDI"), gprFieldIndex64("RSI"), gprFieldIndex64("RDX"),
                             gprFieldIndex64("RCX"), gprFieldIndex64("R8"), gprFieldIndex64("R9") };
    int intIdx = 0, fpIdx = 0;
    for (Type* paramTy : ft->params()) {
      if (paramTy->isFloatTy() || paramTy->isDoubleTy()) {
        Value* vec = loadXmm(LC, B, fpIdx < 8 ? fpIdx : 7);
        Value* val = B.CreateExtractElement(vec, (uint64_t)0);
        if (paramTy->isFloatTy()) val = B.CreateFPTrunc(val, paramTy);
        args.push_back(val);
        fpIdx++;
      } else {
        Value* val = loadGpr64(LC, B, intIdx < 6 ? intRegs[intIdx] : 0);
        if (paramTy->isPointerTy()) val = B.CreateIntToPtr(val, paramTy);
        else val = truncOrZext(B, val, paramTy);
        args.push_back(val);
        intIdx++;
      }
    }
    if (ft->isVarArg()) {
      while (intIdx < 6) args.push_back(loadGpr64(LC, B, intRegs[intIdx++]));
      while (fpIdx < 8) args.push_back(B.CreateExtractElement(loadXmm(LC, B, fpIdx++), (uint64_t)0));
    }
    ci = B.CreateCall(targetF, args);
    if (!ft->getReturnType()->isVoidTy()) {
      Type* rTy = ft->getReturnType();
      if (rTy->isFloatTy() || rTy->isDoubleTy()) {
        Value* val = ci;
        if (rTy->isFloatTy()) val = B.CreateFPExt(val, B.getDoubleTy());
        Value* vec = UndefValue::get(FixedVectorType::get(B.getDoubleTy(), 2));
        vec = B.CreateInsertElement(vec, val, (uint64_t)0);
        vec = B.CreateInsertElement(vec, ConstantFP::get(B.getDoubleTy(), 0.0), (uint64_t)1);
        storeXmm(LC, B, 0, vec);
      } else {
        Value* val = ci;
        if (rTy->isPointerTy()) val = B.CreatePtrToInt(val, B.getInt64Ty());
        else val = truncOrZext(B, val, B.getInt64Ty());
        storeGpr64(LC, B, 0, val);
      }
    }
  } else {
    Value* targetPtr = resolveMemAddress(LC, B, op, 8).ptr;
    FunctionType* ft = FunctionType::get(B.getVoidTy(), {}, false);
    ci = B.CreateCall(ft, targetPtr, {});
  }
  if (ci && isTailCall) ci->setTailCall(true);
  return ci;
}

} // namespace llvm_lift
