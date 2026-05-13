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
- IDIV (32-bit EDX:EAX ÷ r/m32 form)

Comparison / Flags
- CMP
- TEST
- CMOVE
- SETcc family: SETC/SETB, SETE/SETZ, SETNE, SETPE/SETP

Shifts
- SHL
- SHR
- SAR
*/

#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <system_error>
#include <vector>

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>

#include "llvmLiftShared.hpp"
#include <google/protobuf/util/json_util.h>

using namespace llvm;
namespace lift = llvm_lift;

// =============================================================================
// Step-11-specific context / stack promotion
// =============================================================================

struct StackSlotInfo {
  std::string name;
  std::string baseReg;   // normalized uppercase, typically RBP or RSP
  int64_t startOff = 0;
  uint32_t size = 0;
  unsigned align = 1;
  AllocaInst* alloca = nullptr;
};

struct Step11LowerCtx : public lift::FnLowerCtx {
  std::vector<StackSlotInfo> promotedSlots;
  std::map<std::string, std::string>& bbLlvmMapping;

  Step11LowerCtx(LLVMContext& c, Module& m, StructType* stateTy,
                 Function* f, Value* stateArg,
                 const lifted_ast::Program* p, const lifted_ast::Function* fnAst,
                 std::map<std::string, std::string>& bbMap,
                 std::map<std::string, std::string>& instrMap)
      : lift::FnLowerCtx(c, m, stateTy, f, stateArg, p, fnAst, instrMap),
        bbLlvmMapping(bbMap) {}
};

static unsigned normalizeAlignment(int32_t a) {
  unsigned v = (unsigned)std::max<int32_t>(1, a);
  unsigned p2 = 1;
  while ((p2 << 1) > p2 && (p2 << 1) <= v) p2 <<= 1;
  return p2;
}

static Align alignForOffset(unsigned baseAlign, uint64_t offset) {
  uint64_t a = std::max<uint64_t>(1, baseAlign);
  if (offset == 0) return Align(a);
  uint64_t lowbit = offset & (~offset + 1);
  if (lowbit == 0) return Align(1);
  return Align(std::max<uint64_t>(1, std::min<uint64_t>(a, lowbit)));
}

static unsigned memSizeBytesFromOperand(const lifted_ast::Operand& op) {
  if (!op.has_size()) return 0;
  std::string s = lift::toUpper(op.size());
  if (s == "BYTE") return 1;
  if (s == "WORD") return 2;
  if (s == "DWORD") return 4;
  if (s == "QWORD") return 8;
  return 0;
}

static Type* stackSlotAllocaType(LLVMContext& C, int32_t size) {
  uint64_t n = (uint64_t)std::max<int32_t>(1, size);
  return ArrayType::get(Type::getInt8Ty(C), n);
}

static std::optional<std::pair<Value*, Align>>
lookupStackSlotPtr(IRBuilder<>& B,
                   const std::vector<StackSlotInfo>& slots,
                   const std::string& baseRegUpper,
                   int64_t disp,
                   unsigned accessSize) {
  for (const auto& s : slots) {
    if (s.baseReg != baseRegUpper) continue;

    int64_t begin = s.startOff;
    int64_t end = s.startOff + (int64_t)s.size;
    int64_t reqEnd = disp + (int64_t)accessSize;

    if (disp >= begin && reqEnd <= end) {
      uint64_t innerOff = (uint64_t)(disp - begin);
      Value* p = B.CreateGEP(B.getInt8Ty(), s.alloca, B.getInt64(innerOff), "stk.gep");
      return std::make_pair(p, alignForOffset(s.align, innerOff));
    }
  }
  return std::nullopt;
}

static void promoteStackSlots(Step11LowerCtx& LC, IRBuilder<>& EntryB) {
  for (const auto& ss : LC.FnAst->stack_slots()) {
    StackSlotInfo info;
    info.name = ss.has_name() && !ss.name().empty() ? ss.name() : "slot";
    info.baseReg = lift::toUpper(ss.has_register_() ? ss.register_() : "RBP");
    info.startOff = ss.has_offset() ? ss.offset() : 0;
    info.size = (uint32_t)std::max<int32_t>(1, ss.has_size() ? ss.size() : 8);
    info.align = normalizeAlignment(ss.has_alignment() ? ss.alignment() : 1);

    AllocaInst* a = EntryB.CreateAlloca(stackSlotAllocaType(LC.C, (int32_t)info.size), nullptr, info.name);
    a->setAlignment(Align(info.align));
    info.alloca = a;

    LC.promotedSlots.push_back(info);
  }
}

// =============================================================================
// Step-11-specific symbol / address wrappers
// =============================================================================

static Value* getSymbolAddressAsPtr(Step11LowerCtx& LC, IRBuilder<>& B,
                                    const std::string& sym,
                                    const std::string& astInstrId = "") {
  auto it = LC.P->symbol_table().find(sym);
  if (it != LC.P->symbol_table().end()) {
    const auto& se = it->second;
    if (se.has_kind() && se.kind() == "label" && se.has_definition()) {
      if (auto bbIdOpt = lift::structStringField(se.definition(), "bb_id")) {
        auto itBB = LC.bbIdToLlvm.find(*bbIdOpt);
        if (itBB != LC.bbIdToLlvm.end())
          return BlockAddress::get(LC.F, itBB->second);
      }
    }
  }
  return lift::symbolAddressAsPtr(LC, B, sym, astInstrId);
}

static Value* resolveGotSlot(Step11LowerCtx& LC, const std::string& sym) {
  std::string gotName = sym + "@GOT";
  GlobalVariable* gotSlot = LC.M.getNamedGlobal(gotName);
  if (gotSlot) return gotSlot;

  Type* ptrTy = PointerType::getUnqual(LC.C);
  Constant* init = UndefValue::get(ptrTy);

  if (GlobalVariable* gv = LC.M.getNamedGlobal(sym)) {
    init = gv;
  } else if (Function* fn = LC.M.getFunction(sym)) {
    init = fn;
  }

  if (init->getType() != ptrTy)
    init = ConstantExpr::getPointerCast(init, ptrTy);

  gotSlot = new GlobalVariable(LC.M, ptrTy, true,
                               GlobalValue::PrivateLinkage,
                               init, gotName);
  return gotSlot;
}

static lift::MemAddr resolveMemAddressStep11(Step11LowerCtx& LC, IRBuilder<>& B,
                                             const lifted_ast::Operand& op,
                                             unsigned accessSize = 0,
                                             const std::string& astInstrId = "") {
  lift::MemAddr out;
  out.ptr = UndefValue::get(PointerType::getUnqual(LC.C));
  out.align = Align(1);
  out.isSymbolic = false;

  if (!op.has_memory()) return out;
  const auto& m = op.memory();

  const std::string base = m.has_base() ? lift::toUpper(m.base()) : "";

  // Stack-slot promotion first.
  if ((base == "RBP" || base == "RSP") && m.has_displacement()) {
    if (auto dispOpt = lift::valueAsInt64(m.displacement())) {
      if (auto stk = lookupStackSlotPtr(B, LC.promotedSlots, base, *dispOpt, accessSize)) {
        out.ptr = stk->first;
        out.align = stk->second;
        return out;
      }
    }
  }

  // Step-11-specific symbolic / GOT-aware handling.
  if (op.has_symbol_ref()) {
    bool isGot = (op.has_via_got() && op.via_got()) || op.symbol_ref() == "stderr";
    Value* basePtr = isGot
        ? resolveGotSlot(LC, op.symbol_ref())
        : getSymbolAddressAsPtr(LC, B, op.symbol_ref(), astInstrId);

    out.isSymbolic = true;
    out.symName = op.symbol_ref();

    if (m.has_displacement()) {
      if (auto dispOpt = lift::valueAsInt64(m.displacement()); dispOpt && *dispOpt != 0) {
        basePtr = B.CreateGEP(B.getInt8Ty(), basePtr, B.getInt64(*dispOpt),
                              isGot ? "got.disp" : "sym.disp");
        if (auto* I = dyn_cast<Instruction>(basePtr)) {
          lift::attachPicRelocations(I, out.symName, LC.C);
          if (!astInstrId.empty()) lift::attachAstInstrId(I, astInstrId, LC.C);
        }
      }
    }

    out.ptr = basePtr;
    return out;
  }

  // RIP-relative symbolic label/global/function reference.
  if (base == "RIP" && m.has_displacement()) {
    const auto& disp = m.displacement();
    std::string sym;

    if (disp.kind_case() == google::protobuf::Value::kStringValue) {
      sym = disp.string_value();
    } else if (auto sOpt = lift::structStringField(disp, "symbol")) {
      sym = *sOpt;
    }

    if (!sym.empty() &&
        (LC.M.getNamedGlobal(sym) || LC.M.getFunction(sym) || LC.P->symbol_table().count(sym))) {
      out.ptr = getSymbolAddressAsPtr(LC, B, sym, astInstrId);
      out.isSymbolic = true;
      out.symName = sym;
      return out;
    }
  }

  // Defer all remaining generic resolution to the shared authoritative helper.
  return lift::resolveMemAddress(LC, B, op, accessSize, astInstrId);
}

// =============================================================================
// Local helpers
// =============================================================================

static void recordMapping(Step11LowerCtx& LC, const lifted_ast::Instruction& insn, Value* v) {
  if (!insn.has_id() || insn.id().empty() || !v) return;

  std::string name = "instr_" + insn.id();

  if (auto* I = dyn_cast<Instruction>(v)) {
    if (!I->getType()->isVoidTy()) I->setName(name);
    lift::attachAstInstrId(I, insn.id(), LC.C);
  }

  LC.instrLlvmMapping[insn.id()] = name;
}

static void writeGprSubreg(Step11LowerCtx& LC, IRBuilder<>& B,
                           const lift::RegInfo& ri, Value* v) {
  if (!ri.isValid || ri.isXmm || ri.gprIndex < 0) return;

  Type* i64 = B.getInt64Ty();
  unsigned w = ri.bitWidth;
  unsigned off = ri.bitOffset;

  if (w == 64 && off == 0) {
    Value* vi64 = v->getType()->isPointerTy() ? B.CreatePtrToInt(v, i64)
                                              : lift::truncOrZext(B, v, i64);
    lift::storeGpr64(LC, B, (unsigned)ri.gprIndex, vi64);
    return;
  }

  if (w == 32 && off == 0) {
    Value* vi32 = lift::truncOrZext(B, v, B.getInt32Ty());
    Value* z = B.CreateZExt(vi32, i64, "zext32");
    lift::storeGpr64(LC, B, (unsigned)ri.gprIndex, z);
    return;
  }

  Value* old = lift::loadGpr64(LC, B, (unsigned)ri.gprIndex);
  Type* subTy = lift::intTy(LC.C, w);
  Value* sub = lift::truncOrZext(B, v, subTy);
  Value* sub64 = B.CreateZExt(sub, i64, "sub.zext64");
  if (off != 0) sub64 = B.CreateShl(sub64, B.getInt64(off), "sub.shl");

  uint64_t mask = (w == 64) ? ~0ULL : ((1ULL << w) - 1ULL);
  mask <<= off;

  Value* maskV = ConstantInt::get(i64, mask);
  Value* invMaskV = ConstantInt::get(i64, ~mask);
  Value* kept = B.CreateAnd(old, invMaskV, "merge.kept");
  Value* ins = B.CreateAnd(sub64, maskV, "merge.ins");
  Value* merged = B.CreateOr(kept, ins, "merge");

  lift::storeGpr64(LC, B, (unsigned)ri.gprIndex, merged);
}

static Value* evalExprToI64(Step11LowerCtx& LC, IRBuilder<>& B,
                            const google::protobuf::Value& v,
                            const std::string& astInstrId = "") {
  using V = google::protobuf::Value;

  if (auto i = lift::valueAsInt64(v)) return B.getInt64(*i);

  if (v.kind_case() == V::kStructValue) {
    if (auto regOpt = lift::structStringField(v, "register")) {
      lift::RegInfo ri = lift::decodeReg(*regOpt);
      if (ri.isValid && !ri.isXmm) {
        Value* x = lift::readGprSubreg(LC, B, ri);
        return lift::truncOrZext(B, x, B.getInt64Ty());
      }
    }

    if (auto symOpt = lift::structStringField(v, "symbol")) {
      const std::string sym = *symOpt;

      if (GlobalVariable* GV = LC.M.getNamedGlobal(sym)) {
        if (GV->isConstant() && GV->hasInitializer()) {
          if (auto* CI = dyn_cast<ConstantInt>(GV->getInitializer()))
            return lift::truncOrZext(B, CI, B.getInt64Ty());
        }
      }

      Value* p = getSymbolAddressAsPtr(LC, B, sym, astInstrId);
      Value* pi64 = B.CreatePtrToInt(p, B.getInt64Ty(), "sym.ptrtoint");
      if (auto* I = dyn_cast<Instruction>(pi64))
        if (!astInstrId.empty()) lift::attachAstInstrId(I, astInstrId, LC.C);
      return pi64;
    }

    if (auto addV = lift::structFieldValue(v, "additive")) {
      if (addV->kind_case() == V::kListValue) {
        Value* acc = B.getInt64(0);
        for (const auto& elt : addV->list_value().values())
          acc = B.CreateAdd(acc, evalExprToI64(LC, B, elt, astInstrId), "add");
        return acc;
      }
    }

    if (auto subV = lift::structFieldValue(v, "subtract")) {
      if (subV->kind_case() == V::kListValue) {
        const auto& xs = subV->list_value().values();
        if (xs.empty()) return B.getInt64(0);
        Value* acc = evalExprToI64(LC, B, xs[0], astInstrId);
        for (int i = 1; i < xs.size(); ++i)
          acc = B.CreateSub(acc, evalExprToI64(LC, B, xs[i], astInstrId), "sub");
        return acc;
      }
    }
  }

  return UndefValue::get(B.getInt64Ty());
}

static Value* loadFromMem(Step11LowerCtx& LC, IRBuilder<>& B,
                          const lifted_ast::Operand& memOp, Type* ty,
                          const std::string& astInstrId = "") {
  unsigned accessSize = 8;
  if (ty->isIntegerTy()) accessSize = ty->getIntegerBitWidth() / 8;
  else if (ty->isPointerTy()) accessSize = 8;

  lift::MemAddr a = resolveMemAddressStep11(LC, B, memOp, accessSize, astInstrId);
  LoadInst* L = B.CreateLoad(ty, a.ptr, "mem.ld");
  if (a.align.value() > 1) L->setAlignment(a.align);

  if (a.isSymbolic && !a.symName.empty()) lift::attachPicRelocations(L, a.symName, LC.C);
  if (!astInstrId.empty()) lift::attachAstInstrId(L, astInstrId, LC.C);
  return L;
}

static void storeToMem(Step11LowerCtx& LC, IRBuilder<>& B,
                       const lifted_ast::Operand& memOp, Value* v, Type* storeTy,
                       const std::string& astInstrId = "") {
  unsigned accessSize = 8;
  if (storeTy->isIntegerTy()) accessSize = storeTy->getIntegerBitWidth() / 8;
  else if (storeTy->isPointerTy()) accessSize = 8;

  lift::MemAddr a = resolveMemAddressStep11(LC, B, memOp, accessSize, astInstrId);
  Value* vv = v;

  if (vv->getType() != storeTy) {
    if (storeTy->isIntegerTy()) vv = lift::truncOrZext(B, vv, storeTy);
    else if (storeTy->isPointerTy() && vv->getType()->isIntegerTy(64)) vv = B.CreateIntToPtr(vv, storeTy);
    else if (storeTy->isIntegerTy(64) && vv->getType()->isPointerTy()) vv = B.CreatePtrToInt(vv, storeTy);
    else vv = B.CreateBitCast(vv, storeTy);
  }

  StoreInst* S = B.CreateStore(vv, a.ptr);
  if (a.align.value() > 1) S->setAlignment(a.align);

  if (a.isSymbolic && !a.symName.empty()) lift::attachPicRelocations(S, a.symName, LC.C);
  if (!astInstrId.empty()) lift::attachAstInstrId(S, astInstrId, LC.C);
}

static Value* resolveRValue(Step11LowerCtx& LC, IRBuilder<>& B,
                            const lifted_ast::Operand& op, Type* desiredTy,
                            const std::string& astInstrId = "") {
  // 1. Registers
  if (op.has_register_()) {
    lift::RegInfo ri = lift::decodeReg(op.register_());

    if (ri.isValid && ri.isXmm) {
      Type* xmmTy = FixedVectorType::get(B.getDoubleTy(), 2);
      Value* v = lift::loadXmm(LC, B, ri.xmmIndex);
      if (desiredTy == xmmTy) return v;
      return UndefValue::get(desiredTy);
    }

    if (ri.isValid && !ri.isXmm) {
      Value* v = lift::readGprSubreg(LC, B, ri);
      if (desiredTy->isIntegerTy()) return lift::truncOrZext(B, v, desiredTy);
      if (desiredTy->isPointerTy()) {
        Value* asI64 = lift::truncOrZext(B, v, B.getInt64Ty());
        return B.CreateIntToPtr(asI64, desiredTy, "reg.inttoptr");
      }
      return UndefValue::get(desiredTy);
    }
  }

  // 2. Memory
  if (op.has_memory()) return loadFromMem(LC, B, op, desiredTy, astInstrId);

  // 3. Symbolic immediates
  if (op.has_symbol_ref()) {
    Value* val = nullptr;
    const std::string sym = op.symbol_ref();

    if (GlobalVariable* GV = LC.M.getNamedGlobal(sym)) {
      if (GV->isConstant() && GV->hasInitializer()) {
        if (auto* CI = dyn_cast<ConstantInt>(GV->getInitializer()))
          val = CI;
      }
    }

    if (!val) {
      Value* ptr = getSymbolAddressAsPtr(LC, B, sym, astInstrId);
      val = B.CreatePtrToInt(ptr, B.getInt64Ty(), "sym.imm.ptrtoint");
      if (auto* I = dyn_cast<Instruction>(val))
        if (!astInstrId.empty()) lift::attachAstInstrId(I, astInstrId, LC.C);
    }

    val = lift::truncOrZext(B, val, B.getInt64Ty());

    int64_t addend = 0;
    if (op.has_integer() && op.integer().has_value()) {
      if (auto iOpt = lift::valueAsInt64(op.integer().value())) addend = *iOpt;
    }
    if (addend != 0)
      val = B.CreateAdd(val, B.getInt64(addend), "sym.imm.add");

    if (desiredTy->isIntegerTy()) return lift::truncOrZext(B, val, desiredTy);
    if (desiredTy->isPointerTy()) return B.CreateIntToPtr(val, desiredTy, "sym.imm.inttoptr");

    if (desiredTy->isFloatingPointTy()) {
      unsigned dstBits = desiredTy->getPrimitiveSizeInBits();
      if (dstBits > 0)
        return B.CreateBitCast(lift::truncOrZext(B, val, lift::intTy(LC.C, dstBits)), desiredTy);
    }

    return UndefValue::get(desiredTy);
  }

  // 4. Expression immediates
  if (op.has_expression()) {
    Value* exprI64 = evalExprToI64(LC, B, op.expression(), astInstrId);
    if (desiredTy->isIntegerTy()) return lift::truncOrZext(B, exprI64, desiredTy);
    if (desiredTy->isPointerTy()) return B.CreateIntToPtr(exprI64, desiredTy, "expr.inttoptr");
    return UndefValue::get(desiredTy);
  }

  // 5. Integer immediates
  if (op.has_integer()) {
    int64_t imm = 0;
    if (op.integer().has_value()) {
      if (auto iOpt = lift::valueAsInt64(op.integer().value())) imm = *iOpt;
    }

    if (desiredTy->isIntegerTy()) {
      unsigned bits = desiredTy->getIntegerBitWidth();
      APInt api(bits, (uint64_t)imm, /*isSigned=*/true);
      return ConstantInt::get(desiredTy, api);
    }

    if (desiredTy->isPointerTy()) {
      Value* ci = ConstantInt::get(B.getInt64Ty(), (uint64_t)imm, true);
      return B.CreateIntToPtr(ci, desiredTy, "imm.inttoptr");
    }

    if (desiredTy->isFloatingPointTy())
      return ConstantFP::get(desiredTy, (double)imm);

    return UndefValue::get(desiredTy);
  }

  return UndefValue::get(desiredTy);
}

// =============================================================================
// Flags / type helpers
// =============================================================================

static Value* computePF(IRBuilder<>& B, Value* resIntN) {
  Value* lo8 = B.CreateTrunc(resIntN, B.getInt8Ty(), "pf.lo8");
  Function* ctpop = Intrinsic::getDeclaration(
      B.GetInsertBlock()->getModule(), Intrinsic::ctpop, {B.getInt8Ty()});
  Value* pop = B.CreateCall(ctpop, {lo8}, "pf.pop");
  Value* lsb = B.CreateAnd(pop, B.getInt8(1), "pf.lsb");
  Value* odd = B.CreateICmpEQ(lsb, B.getInt8(1), "pf.odd");
  return B.CreateNot(odd, "pf");
}

static void updateFlagsLogic(Step11LowerCtx& LC, IRBuilder<>& B, Value* res) {
  lift::storeFlag(LC, B, lift::CF, B.getFalse());
  lift::storeFlag(LC, B, lift::OF, B.getFalse());
  lift::storeFlag(LC, B, lift::AF, B.getFalse());
  lift::storeFlag(LC, B, lift::ZF,
                  B.CreateICmpEQ(res, ConstantInt::get(res->getType(), 0), "zf"));
  lift::storeFlag(LC, B, lift::SF,
                  B.CreateICmpSLT(res, ConstantInt::get(res->getType(), 0), "sf"));
  lift::storeFlag(LC, B, lift::PF, computePF(B, res));
}

static void updateFlagsAddSubCommon(Step11LowerCtx& LC, IRBuilder<>& B,
                                    Value* a, Value* b, Value* res,
                                    Value* cf, Value* of) {
  lift::storeFlag(LC, B, lift::ZF,
                  B.CreateICmpEQ(res, ConstantInt::get(res->getType(), 0), "zf"));
  lift::storeFlag(LC, B, lift::SF,
                  B.CreateICmpSLT(res, ConstantInt::get(res->getType(), 0), "sf"));
  lift::storeFlag(LC, B, lift::PF, computePF(B, res));

  Value* x = B.CreateXor(a, b, "af.x1");
  x = B.CreateXor(x, res, "af.x2");
  Value* mask = ConstantInt::get(res->getType(), 0x10);
  lift::storeFlag(LC, B, lift::AF,
                  B.CreateICmpNE(B.CreateAnd(x, mask),
                                 ConstantInt::get(res->getType(), 0), "af"));
  lift::storeFlag(LC, B, lift::CF, cf);
  lift::storeFlag(LC, B, lift::OF, of);
}

static void updateFlagsAdd(Step11LowerCtx& LC, IRBuilder<>& B,
                           Value* a, Value* b, Value* res) {
  Type* ty = res->getType();
  Function* uadd = Intrinsic::getDeclaration(&LC.M, Intrinsic::uadd_with_overflow, {ty});
  Function* sadd = Intrinsic::getDeclaration(&LC.M, Intrinsic::sadd_with_overflow, {ty});
  Value* u = B.CreateCall(uadd, {a, b}, "uadd.ov");
  Value* s = B.CreateCall(sadd, {a, b}, "sadd.ov");
  updateFlagsAddSubCommon(LC, B, a, b, res,
                          B.CreateExtractValue(u, 1, "cf"),
                          B.CreateExtractValue(s, 1, "of"));
}

static void updateFlagsSub(Step11LowerCtx& LC, IRBuilder<>& B,
                           Value* a, Value* b, Value* res) {
  Type* ty = res->getType();
  Function* usub = Intrinsic::getDeclaration(&LC.M, Intrinsic::usub_with_overflow, {ty});
  Function* ssub = Intrinsic::getDeclaration(&LC.M, Intrinsic::ssub_with_overflow, {ty});
  Value* u = B.CreateCall(usub, {a, b}, "usub.ov");
  Value* s = B.CreateCall(ssub, {a, b}, "ssub.ov");
  updateFlagsAddSubCommon(LC, B, a, b, res,
                          B.CreateExtractValue(u, 1, "cf"),
                          B.CreateExtractValue(s, 1, "of"));
}

static void storeFlagIfNonZeroShift(Step11LowerCtx& LC, IRBuilder<>& B,
                                    lift::FlagIndex f, Value* doUpdate, Value* newVal) {
  Value* merged = B.CreateSelect(doUpdate, newVal, lift::loadFlag(LC, B, f));
  lift::storeFlag(LC, B, f, merged);
}

static void updateShiftStatusFlags(Step11LowerCtx& LC, IRBuilder<>& B,
                                   Value* doUpdate, Value* res) {
  storeFlagIfNonZeroShift(
      LC, B, lift::ZF, doUpdate,
      B.CreateICmpEQ(res, ConstantInt::get(res->getType(), 0), "zf"));
  storeFlagIfNonZeroShift(
      LC, B, lift::SF, doUpdate,
      B.CreateICmpSLT(res, ConstantInt::get(res->getType(), 0), "sf"));
  storeFlagIfNonZeroShift(
      LC, B, lift::PF, doUpdate,
      computePF(B, res));
}

static bool isTerminatorOpcode(const std::string& opcUpper) {
  static const std::set<std::string> terms = {
    "RET","JMP","JE","JNE","JL","JLE","JG","JGE","JA","JAE","JB","JBE",
    "JO","JNO","JS","JNS","JP","JNP","JC","JNC",
    "LOOP","LOOPE","LOOPNE","IRET","SYSRET"
  };
  return terms.count(opcUpper) != 0;
}

static unsigned regWidthBits(const std::string& reg) {
  lift::RegInfo ri = lift::decodeReg(reg);
  if (!ri.isValid || ri.isXmm) return 0;
  return ri.bitWidth;
}

static Type* chooseOpIntType(Step11LowerCtx& LC, const lifted_ast::Instruction& insn) {
  if (insn.has_op_refinement()) {
    std::string r = lift::toUpper(insn.op_refinement());
    if (r == "I8" || r.rfind("I8", 0) == 0 || r == "CHAR") return Type::getInt8Ty(LC.C);
    if (r == "I16" || r.rfind("I16", 0) == 0) return Type::getInt16Ty(LC.C);
    if (r == "I32" || r.rfind("I32", 0) == 0) return Type::getInt32Ty(LC.C);
    if (r == "I64" || r.rfind("I64", 0) == 0 || r == "PTR") return Type::getInt64Ty(LC.C);
  }
  return Type::getInt64Ty(LC.C);
}

static Type* inferIntTypeFromOperand(Step11LowerCtx& LC,
                                     const lifted_ast::Operand& op,
                                     const lifted_ast::Instruction& insn) {
  Type* ty = nullptr;
  if (op.has_register_()) {
    unsigned w = regWidthBits(op.register_());
    ty = lift::intTy(LC.C, w == 0 ? 64 : w);
  } else if (op.has_memory()) {
    unsigned sz = memSizeBytesFromOperand(op);
    ty = sz == 0 ? chooseOpIntType(LC, insn) : lift::intTy(LC.C, sz * 8);
  } else {
    ty = chooseOpIntType(LC, insn);
  }
  if (!ty || !ty->isIntegerTy()) ty = Type::getInt64Ty(LC.C);
  return ty;
}

static Value* resolveShiftCount(Step11LowerCtx& LC, IRBuilder<>& B,
                                const lifted_ast::Operand& cnt, Type* ty,
                                const std::string& astInstrId) {
  if (cnt.has_integer() && cnt.integer().has_value()) {
    int64_t imm = lift::valueAsInt64(cnt.integer().value()).value_or(0);
    return ConstantInt::get(ty, (uint64_t)imm);
  }

  if (cnt.has_register_()) {
    lift::RegInfo ri = lift::decodeReg(cnt.register_());
    if (ri.isValid && !ri.isXmm)
      return lift::truncOrZext(B, lift::readGprSubreg(LC, B, ri), ty);
  }

  (void)astInstrId;
  return ConstantInt::get(ty, 0);
}

// =============================================================================
// Lowering
// =============================================================================

static void lowerMOV(Step11LowerCtx& LC, IRBuilder<>& B,
                     const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 2) return;

  const auto& dst = insn.operands(0);
  const auto& src = insn.operands(1);
  Type* ty = inferIntTypeFromOperand(LC, dst, insn);

  Value* v = resolveRValue(LC, B, src, ty, insn.id());

  if (dst.has_register_()) {
    lift::RegInfo dri = lift::decodeReg(dst.register_());
    if (!dri.isValid || dri.isXmm) return;
    writeGprSubreg(LC, B, dri, v);
    recordMapping(LC, insn, v);
  } else if (dst.has_memory()) {
    storeToMem(LC, B, dst, v, ty, insn.id());
    recordMapping(LC, insn, v);
  }
}

static void lowerLEA(Step11LowerCtx& LC, IRBuilder<>& B,
                     const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 2) return;

  const auto& dst = insn.operands(0);
  const auto& src = insn.operands(1);

  if (!dst.has_register_()) return;
  lift::RegInfo dri = lift::decodeReg(dst.register_());
  if (!dri.isValid || dri.isXmm) return;

  Value* addrI64 = nullptr;

  if (src.has_memory()) {
    lift::MemAddr a = resolveMemAddressStep11(LC, B, src, 1, insn.id());
    Value* pi64 = B.CreatePtrToInt(a.ptr, B.getInt64Ty(), "lea.ptrtoint");
    if (auto* I = dyn_cast<Instruction>(pi64)) lift::attachAstInstrId(I, insn.id(), LC.C);
    addrI64 = pi64;
  } else if (src.has_symbol_ref()) {
    Value* p = getSymbolAddressAsPtr(LC, B, src.symbol_ref(), insn.id());
    Value* pi64 = B.CreatePtrToInt(p, B.getInt64Ty(), "lea.sym");
    if (auto* I = dyn_cast<Instruction>(pi64)) lift::attachAstInstrId(I, insn.id(), LC.C);

    int64_t addend = 0;
    if (src.has_integer() && src.integer().has_value()) {
      if (auto iOpt = lift::valueAsInt64(src.integer().value())) addend = *iOpt;
    }
    if (addend != 0)
      pi64 = B.CreateAdd(pi64, B.getInt64(addend), "lea.sym.add");

    addrI64 = pi64;
  } else if (src.has_expression()) {
    addrI64 = evalExprToI64(LC, B, src.expression(), insn.id());
  } else {
    addrI64 = UndefValue::get(B.getInt64Ty());
  }

  Type* dstTy = lift::intTy(LC.C, dri.bitWidth);
  Value* out = lift::truncOrZext(B, addrI64, dstTy);
  writeGprSubreg(LC, B, dri, out);
  recordMapping(LC, insn, out);
}

static void lowerALU2(Step11LowerCtx& LC, IRBuilder<>& B,
                      const lifted_ast::Instruction& insn,
                      const std::string& opcUpper) {
  if (insn.operands_size() < 2) return;

  const auto& dst = insn.operands(0);
  const auto& src = insn.operands(1);
  Type* ty = inferIntTypeFromOperand(LC, dst, insn);

  Value* a = nullptr;
  if (dst.has_register_()) {
    lift::RegInfo dri = lift::decodeReg(dst.register_());
    if (!dri.isValid || dri.isXmm) return;
    a = lift::truncOrZext(B, lift::readGprSubreg(LC, B, dri), ty);
  } else if (dst.has_memory()) {
    a = loadFromMem(LC, B, dst, ty, insn.id());
  } else {
    return;
  }

  Value* b = resolveRValue(LC, B, src, ty, insn.id());
  Value* res = nullptr;
  bool isLogic = false;

  if (opcUpper == "ADD") {
    res = B.CreateAdd(a, b, "add");
    updateFlagsAdd(LC, B, a, b, res);
  } else if (opcUpper == "SUB") {
    res = B.CreateSub(a, b, "sub");
    updateFlagsSub(LC, B, a, b, res);
  } else if (opcUpper == "XOR") {
    res = B.CreateXor(a, b, "xor");
    isLogic = true;
  } else if (opcUpper == "AND") {
    res = B.CreateAnd(a, b, "and");
    isLogic = true;
  } else if (opcUpper == "OR") {
    res = B.CreateOr(a, b, "or");
    isLogic = true;
  } else {
    return;
  }

  if (isLogic) updateFlagsLogic(LC, B, res);

  if (dst.has_register_()) {
    writeGprSubreg(LC, B, lift::decodeReg(dst.register_()), res);
  } else if (dst.has_memory()) {
    storeToMem(LC, B, dst, res, ty, insn.id());
  }

  recordMapping(LC, insn, res);
}

static void lowerCMP(Step11LowerCtx& LC, IRBuilder<>& B,
                     const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 2) return;

  const auto& aOp = insn.operands(0);
  const auto& bOp = insn.operands(1);
  Type* ty = inferIntTypeFromOperand(LC, aOp, insn);

  Value* a = resolveRValue(LC, B, aOp, ty, insn.id());
  Value* b = resolveRValue(LC, B, bOp, ty, insn.id());
  Value* res = B.CreateSub(a, b, "cmp.sub");

  updateFlagsSub(LC, B, a, b, res);
  recordMapping(LC, insn, res);
}

static void lowerTEST(Step11LowerCtx& LC, IRBuilder<>& B,
                      const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 2) return;

  const auto& aOp = insn.operands(0);
  const auto& bOp = insn.operands(1);
  Type* ty = inferIntTypeFromOperand(LC, aOp, insn);

  Value* a = resolveRValue(LC, B, aOp, ty, insn.id());
  Value* b = resolveRValue(LC, B, bOp, ty, insn.id());
  Value* res = B.CreateAnd(a, b, "test.and");

  updateFlagsLogic(LC, B, res);
  recordMapping(LC, insn, res);
}

static void lowerMOVZX_MOVSX(Step11LowerCtx& LC, IRBuilder<>& B,
                             const lifted_ast::Instruction& insn,
                             bool isSignExtend) {
  if (insn.operands_size() < 2) return;

  const auto& dst = insn.operands(0);
  const auto& src = insn.operands(1);

  if (!dst.has_register_()) return;
  lift::RegInfo dri = lift::decodeReg(dst.register_());
  if (!dri.isValid || dri.isXmm) return;

  Type* dstTy = lift::intTy(LC.C, dri.bitWidth);
  Type* srcTy = nullptr;

  if (src.has_register_()) {
    unsigned sw = regWidthBits(src.register_());
    srcTy = lift::intTy(LC.C, sw == 0 ? 8 : sw);
  } else if (src.has_memory()) {
    unsigned sz = memSizeBytesFromOperand(src);
    srcTy = lift::intTy(LC.C, sz == 0 ? 8 : (sz * 8));
  } else {
    srcTy = dstTy;
  }

  Value* v = resolveRValue(LC, B, src, srcTy, insn.id());
  Value* ext = isSignExtend ? lift::truncOrSext(B, v, dstTy)
                            : lift::truncOrZext(B, v, dstTy);
  writeGprSubreg(LC, B, dri, ext);
  recordMapping(LC, insn, ext);
}

static void lowerSETcc(Step11LowerCtx& LC, IRBuilder<>& B,
                       const lifted_ast::Instruction& insn,
                       const std::string& opcUpper) {
  if (insn.operands_size() < 1) return;

  const auto& dst = insn.operands(0);
  Value* cond = nullptr;

  if (opcUpper == "SETC" || opcUpper == "SETB") cond = lift::loadFlag(LC, B, lift::CF);
  else if (opcUpper == "SETNE") cond = B.CreateNot(lift::loadFlag(LC, B, lift::ZF), "setne");
  else if (opcUpper == "SETE" || opcUpper == "SETZ") cond = lift::loadFlag(LC, B, lift::ZF);
  else if (opcUpper == "SETPE" || opcUpper == "SETP") cond = lift::loadFlag(LC, B, lift::PF);
  else cond = B.getFalse();

  Value* byteV = B.CreateZExt(cond, B.getInt8Ty(), "setcc.i8");

  if (dst.has_register_()) {
    writeGprSubreg(LC, B, lift::decodeReg(dst.register_()), byteV);
  } else if (dst.has_memory()) {
    storeToMem(LC, B, dst, byteV, B.getInt8Ty(), insn.id());
  }

  recordMapping(LC, insn, byteV);
}

static void lowerCMOVE(Step11LowerCtx& LC, IRBuilder<>& B,
                       const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 2) return;

  const auto& dst = insn.operands(0);
  const auto& src = insn.operands(1);

  if (!dst.has_register_()) return;
  lift::RegInfo dri = lift::decodeReg(dst.register_());
  if (!dri.isValid || dri.isXmm) return;

  Type* ty = lift::intTy(LC.C, dri.bitWidth);
  Value* oldv = lift::truncOrZext(B, lift::readGprSubreg(LC, B, dri), ty);
  Value* newv = resolveRValue(LC, B, src, ty, insn.id());

  Value* res = B.CreateSelect(lift::loadFlag(LC, B, lift::ZF), newv, oldv, "cmove.sel");
  writeGprSubreg(LC, B, dri, res);
  recordMapping(LC, insn, res);
}

static void lowerINC(Step11LowerCtx& LC, IRBuilder<>& B,
                     const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 1) return;

  const auto& dst = insn.operands(0);
  Type* ty = inferIntTypeFromOperand(LC, dst, insn);

  Value* old = nullptr;
  if (dst.has_register_()) {
    lift::RegInfo ri = lift::decodeReg(dst.register_());
    if (!ri.isValid || ri.isXmm) return;
    old = lift::truncOrZext(B, lift::readGprSubreg(LC, B, ri), ty);
  } else if (dst.has_memory()) {
    old = loadFromMem(LC, B, dst, ty, insn.id());
  } else {
    return;
  }

  Value* one = ConstantInt::get(ty, 1);
  Value* res = B.CreateAdd(old, one, "inc");

  Function* sadd = Intrinsic::getDeclaration(&LC.M, Intrinsic::sadd_with_overflow, {ty});
  Value* s = B.CreateCall(sadd, {old, one}, "inc.ov");
  lift::storeFlag(LC, B, lift::OF, B.CreateExtractValue(s, 1, "of"));
  lift::storeFlag(LC, B, lift::ZF, B.CreateICmpEQ(res, ConstantInt::get(ty, 0), "zf"));
  lift::storeFlag(LC, B, lift::SF, B.CreateICmpSLT(res, ConstantInt::get(ty, 0), "sf"));
  lift::storeFlag(LC, B, lift::PF, computePF(B, res));

  Value* x = B.CreateXor(old, one);
  x = B.CreateXor(x, res);
  lift::storeFlag(LC, B, lift::AF,
                  B.CreateICmpNE(B.CreateAnd(x, ConstantInt::get(ty, 0x10)),
                                 ConstantInt::get(ty, 0)));

  if (dst.has_register_()) {
    writeGprSubreg(LC, B, lift::decodeReg(dst.register_()), res);
  } else {
    storeToMem(LC, B, dst, res, ty, insn.id());
  }

  recordMapping(LC, insn, res);
}

static void lowerSHL(Step11LowerCtx& LC, IRBuilder<>& B,
                     const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 2) return;

  const auto& dst = insn.operands(0);
  const auto& cnt = insn.operands(1);
  Type* ty = inferIntTypeFromOperand(LC, dst, insn);

  Value* old = nullptr;
  if (dst.has_register_()) {
    lift::RegInfo ri = lift::decodeReg(dst.register_());
    if (!ri.isValid || ri.isXmm) return;
    old = lift::truncOrZext(B, lift::readGprSubreg(LC, B, ri), ty);
  } else if (dst.has_memory()) {
    old = loadFromMem(LC, B, dst, ty, insn.id());
  } else {
    return;
  }

  Value* count = resolveShiftCount(LC, B, cnt, ty, insn.id());
  unsigned w = ty->getIntegerBitWidth();
  Value* zero = ConstantInt::get(ty, 0);
  Value* one = ConstantInt::get(ty, 1);
  Value* mask = ConstantInt::get(ty, (uint64_t)(w - 1));
  Value* masked = B.CreateAnd(count, mask, "shl.mask");
  Value* doUpdate = B.CreateICmpNE(masked, zero);
  Value* isOne = B.CreateICmpEQ(masked, one);
  Value* safeMasked = B.CreateSelect(doUpdate, masked, one, "shl.safe");

  Value* res = B.CreateShl(old, masked, "shl");

  Value* cfShiftAmt = B.CreateSub(ConstantInt::get(ty, w), safeMasked, "shl.cf.pos");
  Value* cfWord = B.CreateLShr(old, cfShiftAmt);
  Value* newCf = B.CreateTrunc(B.CreateAnd(cfWord, one), B.getInt1Ty(), "cf");
  storeFlagIfNonZeroShift(LC, B, lift::CF, doUpdate, newCf);

  Value* resMsb = B.CreateLShr(res, ConstantInt::get(ty, w - 1), "shl.res.msb");
  Value* resMsbI1 = B.CreateTrunc(B.CreateAnd(resMsb, one), B.getInt1Ty(), "shl.res.msb.i1");
  Value* ofOne = B.CreateXor(resMsbI1, newCf, "shl.of1");
  Value* of = B.CreateSelect(isOne, ofOne, lift::loadFlag(LC, B, lift::OF));
  lift::storeFlag(LC, B, lift::OF, of);

  updateShiftStatusFlags(LC, B, doUpdate, res);

  if (dst.has_register_()) writeGprSubreg(LC, B, lift::decodeReg(dst.register_()), res);
  else storeToMem(LC, B, dst, res, ty, insn.id());

  recordMapping(LC, insn, res);
}

static void lowerSHR(Step11LowerCtx& LC, IRBuilder<>& B,
                     const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 2) return;

  const auto& dst = insn.operands(0);
  const auto& cnt = insn.operands(1);
  Type* ty = inferIntTypeFromOperand(LC, dst, insn);

  Value* old = nullptr;
  if (dst.has_register_()) {
    lift::RegInfo ri = lift::decodeReg(dst.register_());
    if (!ri.isValid || ri.isXmm) return;
    old = lift::truncOrZext(B, lift::readGprSubreg(LC, B, ri), ty);
  } else if (dst.has_memory()) {
    old = loadFromMem(LC, B, dst, ty, insn.id());
  } else {
    return;
  }

  Value* count = resolveShiftCount(LC, B, cnt, ty, insn.id());
  unsigned w = ty->getIntegerBitWidth();
  Value* zero = ConstantInt::get(ty, 0);
  Value* one = ConstantInt::get(ty, 1);
  Value* mask = ConstantInt::get(ty, (uint64_t)(w - 1));
  Value* masked = B.CreateAnd(count, mask, "shr.mask");
  Value* doUpdate = B.CreateICmpNE(masked, zero);
  Value* isOne = B.CreateICmpEQ(masked, one);
  Value* safeMasked = B.CreateSelect(doUpdate, masked, one, "shr.safe");

  Value* res = B.CreateLShr(old, masked, "shr");

  Value* cfShiftAmt = B.CreateSub(safeMasked, one, "shr.cf.pos");
  Value* tmp = B.CreateLShr(old, cfShiftAmt);
  Value* lastBit = B.CreateAnd(tmp, one);
  Value* newCf = B.CreateTrunc(lastBit, B.getInt1Ty(), "cf");
  storeFlagIfNonZeroShift(LC, B, lift::CF, doUpdate, newCf);

  Value* msb = B.CreateLShr(old, ConstantInt::get(ty, w - 1), "shr.orig.msb");
  Value* ofOne = B.CreateTrunc(B.CreateAnd(msb, one), B.getInt1Ty(), "shr.of1");
  Value* of = B.CreateSelect(isOne, ofOne, lift::loadFlag(LC, B, lift::OF));
  lift::storeFlag(LC, B, lift::OF, of);

  updateShiftStatusFlags(LC, B, doUpdate, res);

  if (dst.has_register_()) writeGprSubreg(LC, B, lift::decodeReg(dst.register_()), res);
  else storeToMem(LC, B, dst, res, ty, insn.id());

  recordMapping(LC, insn, res);
}

static void lowerSAR(Step11LowerCtx& LC, IRBuilder<>& B,
                     const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 2) return;

  const auto& dst = insn.operands(0);
  const auto& cnt = insn.operands(1);
  Type* ty = inferIntTypeFromOperand(LC, dst, insn);

  Value* old = nullptr;
  if (dst.has_register_()) {
    lift::RegInfo ri = lift::decodeReg(dst.register_());
    if (!ri.isValid || ri.isXmm) return;
    old = lift::truncOrZext(B, lift::readGprSubreg(LC, B, ri), ty);
  } else if (dst.has_memory()) {
    old = loadFromMem(LC, B, dst, ty, insn.id());
  } else {
    return;
  }

  Value* count = resolveShiftCount(LC, B, cnt, ty, insn.id());
  unsigned w = ty->getIntegerBitWidth();
  Value* zero = ConstantInt::get(ty, 0);
  Value* one = ConstantInt::get(ty, 1);
  Value* mask = ConstantInt::get(ty, (uint64_t)(w - 1));
  Value* masked = B.CreateAnd(count, mask, "sar.mask");
  Value* doUpdate = B.CreateICmpNE(masked, zero);
  Value* isOne = B.CreateICmpEQ(masked, one);
  Value* safeMasked = B.CreateSelect(doUpdate, masked, one, "sar.safe");

  Value* res = B.CreateAShr(old, masked, "sar");

  Value* cfShiftAmt = B.CreateSub(safeMasked, one, "sar.cf.pos");
  Value* tmp = B.CreateLShr(old, cfShiftAmt);
  Value* lastBit = B.CreateAnd(tmp, one);
  Value* newCf = B.CreateTrunc(lastBit, B.getInt1Ty(), "cf");
  storeFlagIfNonZeroShift(LC, B, lift::CF, doUpdate, newCf);

  Value* of = B.CreateSelect(isOne, B.getFalse(), lift::loadFlag(LC, B, lift::OF));
  lift::storeFlag(LC, B, lift::OF, of);

  updateShiftStatusFlags(LC, B, doUpdate, res);

  if (dst.has_register_()) writeGprSubreg(LC, B, lift::decodeReg(dst.register_()), res);
  else storeToMem(LC, B, dst, res, ty, insn.id());

  recordMapping(LC, insn, res);
}

static void lowerXCHG(Step11LowerCtx& LC, IRBuilder<>& B,
                      const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 2) return;

  const auto& op0 = insn.operands(0);
  const auto& op1 = insn.operands(1);

  const lifted_ast::Operand* memOp = nullptr;
  const lifted_ast::Operand* regOp = nullptr;

  if (op0.has_memory() && op1.has_register_()) {
    memOp = &op0;
    regOp = &op1;
  } else if (op1.has_memory() && op0.has_register_()) {
    memOp = &op1;
    regOp = &op0;
  } else {
    return;
  }

  lift::RegInfo rri = lift::decodeReg(regOp->register_());
  if (!rri.isValid || rri.isXmm) return;

  Type* ty = lift::intTy(LC.C, rri.bitWidth);
  Value* regVal = lift::truncOrZext(B, lift::readGprSubreg(LC, B, rri), ty);

  unsigned accessSize = std::max<unsigned>(1, rri.bitWidth / 8);
  lift::MemAddr ma = resolveMemAddressStep11(LC, B, *memOp, accessSize, insn.id());

  LoadInst* oldMem = B.CreateLoad(ty, ma.ptr, "xchg.ld");
  if (ma.align.value() > 1) oldMem->setAlignment(ma.align);
  if (ma.isSymbolic && !ma.symName.empty()) lift::attachPicRelocations(oldMem, ma.symName, LC.C);
  if (!insn.id().empty()) lift::attachAstInstrId(oldMem, insn.id(), LC.C);

  StoreInst* st = B.CreateStore(regVal, ma.ptr);
  if (ma.align.value() > 1) st->setAlignment(ma.align);
  if (ma.isSymbolic && !ma.symName.empty()) lift::attachPicRelocations(st, ma.symName, LC.C);
  if (!insn.id().empty()) lift::attachAstInstrId(st, insn.id(), LC.C);

  writeGprSubreg(LC, B, rri, oldMem);
  recordMapping(LC, insn, oldMem);
}

static void lowerPOP(Step11LowerCtx& LC, IRBuilder<>& B,
                     const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 1) return;
  const auto& dst = insn.operands(0);
  if (!dst.has_register_()) return;

  lift::RegInfo dri = lift::decodeReg(dst.register_());
  if (!dri.isValid || dri.isXmm) return;

  unsigned rspIdx = (unsigned)lift::gprFieldIndex64("RSP");
  Value* rsp = lift::loadGpr64(LC, B, rspIdx);
  Value* rspPtr = B.CreateIntToPtr(rsp, PointerType::getUnqual(LC.C), "pop.ptr");

  Type* ty = lift::intTy(LC.C, dri.bitWidth);
  LoadInst* loaded = B.CreateLoad(ty, rspPtr, "pop.ld");
  loaded->setAlignment(Align(8));

  writeGprSubreg(LC, B, dri, loaded);

  Value* newRsp = B.CreateAdd(rsp, B.getInt64(8), "pop.rsp");
  lift::storeGpr64(LC, B, rspIdx, newRsp);

  recordMapping(LC, insn, loaded);
}

static void lowerLEAVE(Step11LowerCtx& LC, IRBuilder<>& B,
                       const lifted_ast::Instruction& insn) {
  unsigned rspIdx = (unsigned)lift::gprFieldIndex64("RSP");
  unsigned rbpIdx = (unsigned)lift::gprFieldIndex64("RBP");

  Value* rbpVal = lift::loadGpr64(LC, B, rbpIdx);
  lift::storeGpr64(LC, B, rspIdx, rbpVal);

  Value* rspPtr = B.CreateIntToPtr(rbpVal, PointerType::getUnqual(LC.C), "leave.pop.ptr");
  LoadInst* loaded = B.CreateLoad(B.getInt64Ty(), rspPtr, "leave.pop.ld");
  loaded->setAlignment(Align(8));

  Value* newRsp = B.CreateAdd(rbpVal, B.getInt64(8), "leave.newrsp");

  lift::storeGpr64(LC, B, rbpIdx, loaded);
  lift::storeGpr64(LC, B, rspIdx, newRsp);

  recordMapping(LC, insn, loaded);
}

static void lowerPUSH(Step11LowerCtx& LC, IRBuilder<>& B,
                      const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 1) return;

  const auto& op = insn.operands(0);
  if (!op.has_register_()) return;

  lift::RegInfo ri = lift::decodeReg(op.register_());
  if (!ri.isValid || ri.isXmm) return;

  Value* val = lift::readGprSubreg(LC, B, ri);
  Value* val64 = lift::truncOrZext(B, val, B.getInt64Ty());

  unsigned rspIdx = (unsigned)lift::gprFieldIndex64("RSP");
  Value* rsp = lift::loadGpr64(LC, B, rspIdx);
  Value* newRsp = B.CreateSub(rsp, B.getInt64(8), "push.rsp");
  lift::storeGpr64(LC, B, rspIdx, newRsp);

  Value* rspPtr = B.CreateIntToPtr(newRsp, PointerType::getUnqual(LC.C), "push.ptr");
  StoreInst* st = B.CreateStore(val64, rspPtr);
  st->setAlignment(Align(8));
  if (!insn.id().empty()) lift::attachAstInstrId(st, insn.id(), LC.C);

  recordMapping(LC, insn, newRsp);
}

static void lowerIMUL(Step11LowerCtx& LC, IRBuilder<>& B,
                      const lifted_ast::Instruction& insn) {
  if (insn.operands_size() == 0) return;

  // 1-operand form: RDX:RAX <- RAX * src
  if (insn.operands_size() == 1) {
    const auto& srcOp = insn.operands(0);

    lift::RegInfo raxRi = lift::decodeReg("RAX");
    lift::RegInfo rdxRi = lift::decodeReg("RDX");

    Value* rax = lift::readGprSubreg(LC, B, raxRi);
    Value* src = resolveRValue(LC, B, srcOp, B.getInt64Ty(), insn.id());

    Type* i128 = IntegerType::get(LC.C, 128);
    Value* a128 = B.CreateSExt(rax, i128, "imul.a128");
    Value* b128 = B.CreateSExt(src, i128, "imul.b128");
    Value* prod = B.CreateMul(a128, b128, "imul.full");

    Value* low = B.CreateTrunc(prod, B.getInt64Ty(), "imul.low");
    Value* high = B.CreateTrunc(B.CreateLShr(prod, 64), B.getInt64Ty(), "imul.high");

    writeGprSubreg(LC, B, raxRi, low);
    writeGprSubreg(LC, B, rdxRi, high);

    Value* signExt = B.CreateAShr(low, 63);
    Value* ov = B.CreateICmpNE(high, signExt, "imul.ov");
    lift::storeFlag(LC, B, lift::CF, ov);
    lift::storeFlag(LC, B, lift::OF, ov);

    recordMapping(LC, insn, low);
    return;
  }

  // 2-operand form
  if (insn.operands_size() < 2) return;

  const auto& dst = insn.operands(0);
  const auto& src = insn.operands(1);

  if (!dst.has_register_()) return;
  lift::RegInfo dri = lift::decodeReg(dst.register_());
  if (!dri.isValid || dri.isXmm) return;

  Type* ty = lift::intTy(LC.C, dri.bitWidth);
  Value* a = lift::truncOrZext(B, lift::readGprSubreg(LC, B, dri), ty);
  Value* b = resolveRValue(LC, B, src, ty, insn.id());

  Value* res = B.CreateMul(a, b, "imul.res");

  Function* smul = Intrinsic::getDeclaration(&LC.M, Intrinsic::smul_with_overflow, {ty});
  Value* call = B.CreateCall(smul, {a, b});
  Value* of = B.CreateExtractValue(call, 1);

  lift::storeFlag(LC, B, lift::OF, of);
  lift::storeFlag(LC, B, lift::CF, of);

  writeGprSubreg(LC, B, dri, res);
  recordMapping(LC, insn, res);
}

static void lowerCDQ(Step11LowerCtx& LC, IRBuilder<>& B,
                     const lifted_ast::Instruction& insn) {
  lift::RegInfo eax = lift::decodeReg("EAX");
  lift::RegInfo edx = lift::decodeReg("EDX");
  Value* a = lift::readGprSubreg(LC, B, eax);
  Value* sign = B.CreateAShr(a, B.getInt32(31));
  writeGprSubreg(LC, B, edx, sign);
  recordMapping(LC, insn, sign);
}

static void lowerDIV(Step11LowerCtx& LC, IRBuilder<>& B,
                     const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 1) return;

  const auto& srcOp = insn.operands(0);
  lift::RegInfo eaxRi = lift::decodeReg("EAX");
  lift::RegInfo edxRi = lift::decodeReg("EDX");

  Value* eaxV = lift::readGprSubreg(LC, B, eaxRi);
  Value* edxV = lift::readGprSubreg(LC, B, edxRi);

  Value* hi = B.CreateZExt(edxV, B.getInt64Ty());
  Value* lo = B.CreateZExt(eaxV, B.getInt64Ty());
  Value* dividend = B.CreateOr(B.CreateShl(hi, 32), lo, "div.dividend");

  Value* divisor32 = resolveRValue(LC, B, srcOp, B.getInt32Ty(), insn.id());
  Value* divisor = B.CreateZExt(divisor32, B.getInt64Ty(), "div.divisor");

  Value* q = B.CreateUDiv(dividend, divisor, "div.q");
  Value* r = B.CreateURem(dividend, divisor, "div.r");

  writeGprSubreg(LC, B, eaxRi, B.CreateTrunc(q, B.getInt32Ty()));
  writeGprSubreg(LC, B, edxRi, B.CreateTrunc(r, B.getInt32Ty()));

  recordMapping(LC, insn, q);
}

static void lowerIDIV(Step11LowerCtx& LC, IRBuilder<>& B,
                      const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 1) return;

  const auto& src = insn.operands(0);

  lift::RegInfo eax = lift::decodeReg("EAX");
  lift::RegInfo edx = lift::decodeReg("EDX");

  Value* eaxV = lift::readGprSubreg(LC, B, eax);
  Value* edxV = lift::readGprSubreg(LC, B, edx);

  Value* hi = B.CreateSExt(edxV, B.getInt64Ty(), "idiv.hi.sext");
  Value* lo = B.CreateZExt(eaxV, B.getInt64Ty(), "idiv.lo.zext");
  Value* dividend = B.CreateOr(B.CreateShl(hi, B.getInt64(32)), lo, "idiv.dividend");

  Value* divisor32 = resolveRValue(LC, B, src, B.getInt32Ty(), insn.id());
  Value* divisor = B.CreateSExt(divisor32, B.getInt64Ty(), "idiv.divisor");

  Value* q = B.CreateSDiv(dividend, divisor, "idiv.q");
  Value* r = B.CreateSRem(dividend, divisor, "idiv.r");

  Value* q32 = B.CreateTrunc(q, B.getInt32Ty());
  Value* r32 = B.CreateTrunc(r, B.getInt32Ty());

  writeGprSubreg(LC, B, eax, q32);
  writeGprSubreg(LC, B, edx, r32);

  recordMapping(LC, insn, q32);
}

static void lowerUnsupportedButKeepIR(Step11LowerCtx& LC, IRBuilder<>& B,
                                      const lifted_ast::Instruction& insn) {
  std::string opc = insn.has_opcode() ? lift::toUpper(insn.opcode()) : "UNKNOWN";

  FunctionType* ft = FunctionType::get(Type::getVoidTy(LC.C), false);
  InlineAsm* ia = InlineAsm::get(ft, "; placeholder for " + opc, "", true);
  CallInst* placeholder = B.CreateCall(ia);
  recordMapping(LC, insn, placeholder);

  if (insn.operands_size() >= 1) {
    const auto& dst = insn.operands(0);
    if (dst.has_register_()) {
      lift::RegInfo ri = lift::decodeReg(dst.register_());
      if (ri.isValid) {
        if (ri.isXmm) {
          auto* xmmTy = FixedVectorType::get(B.getDoubleTy(), 2);
          lift::storeXmm(LC, B, ri.xmmIndex, UndefValue::get(xmmTy));
        } else {
          Type* ty = lift::intTy(LC.C, ri.bitWidth);
          writeGprSubreg(LC, B, ri, UndefValue::get(ty));
        }
      }
    }
  }
}

// =============================================================================
// Per-function lowering
// =============================================================================

static void lowerFunction(Step11LowerCtx& LC) {
  auto origLinkage = LC.F->getLinkage();
  LC.F->deleteBody();

  if (LC.F->arg_empty()) {
    errs() << "Warning: lifted function missing %State arg: " << LC.F->getName() << "\n";
    lift::createStubBody(LC.F);
    return;
  }

  LC.StateArg = LC.F->getArg(0);

  for (const auto& bbAst : LC.FnAst->basic_blocks()) {
    std::string bbName =
        (bbAst.has_start_label() && !bbAst.start_label().empty()) ? bbAst.start_label()
        : (bbAst.has_id() && !bbAst.id().empty()) ? bbAst.id()
        : "bb";

    BasicBlock* bb = BasicBlock::Create(LC.C, bbName, LC.F);
    if (bbAst.has_id()) {
      LC.bbIdToLlvm[bbAst.id()] = bb;
      LC.bbLlvmMapping[bbAst.id()] = bb->getName().str();
    }
  }

  if (LC.FnAst->basic_blocks_size() > 0) {
    const auto& entryAst = LC.FnAst->basic_blocks(0);
    auto it = LC.bbIdToLlvm.find(entryAst.id());
    if (it != LC.bbIdToLlvm.end()) {
      IRBuilder<> EntryB(it->second);
      promoteStackSlots(LC, EntryB);
    }
  }

  for (const auto& bbAst : LC.FnAst->basic_blocks()) {
    auto it = LC.bbIdToLlvm.find(bbAst.id());
    if (it == LC.bbIdToLlvm.end()) continue;

    BasicBlock* bb = it->second;
    IRBuilder<> B(bb);
    LC.stateGepCache.clear();

    for (const auto& ie : bbAst.instructions()) {
      const lifted_ast::Instruction& insn = ie.instruction();
      std::string opc = insn.has_opcode() ? lift::toUpper(insn.opcode()) : "";

      if (bbAst.has_terminator() && insn.has_id() && insn.id() == bbAst.terminator()) continue;
      if (isTerminatorOpcode(opc)) continue;

      if (opc == "MOV") lowerMOV(LC, B, insn);
      else if (opc == "LEA") lowerLEA(LC, B, insn);
      else if (opc == "ADD" || opc == "SUB" || opc == "XOR" || opc == "AND" || opc == "OR") lowerALU2(LC, B, insn, opc);
      else if (opc == "CMP") lowerCMP(LC, B, insn);
      else if (opc == "TEST") lowerTEST(LC, B, insn);
      else if (opc == "MOVZX") lowerMOVZX_MOVSX(LC, B, insn, false);
      else if (opc == "MOVSX") lowerMOVZX_MOVSX(LC, B, insn, true);
      else if (opc == "SETC" || opc == "SETB" || opc == "SETNE" ||
               opc == "SETPE" || opc == "SETP" || opc == "SETE" || opc == "SETZ")
        lowerSETcc(LC, B, insn, opc);
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
      else if (opc == "DIV") lowerDIV(LC, B, insn);
      else if (opc == "IDIV") lowerIDIV(LC, B, insn);
      else lowerUnsupportedButKeepIR(LC, B, insn);
    }

    if (!bb->getTerminator()) B.CreateUnreachable();
  }

  LC.F->setLinkage(origLinkage);

  if (verifyFunction(*LC.F, &errs()))
    errs() << "verifyFunction failed for: " << LC.F->getName() << "\n";
}

// =============================================================================
// Module driver
// =============================================================================

static std::unique_ptr<Module> loadBitcodeModule(const std::string& path, LLVMContext& C) {
  auto bufOrErr = MemoryBuffer::getFile(path);
  if (!bufOrErr) {
    std::cerr << "Failed to open bitcode: " << path << "\n";
    return nullptr;
  }

  auto modOrErr = parseBitcodeFile(bufOrErr->get()->getMemBufferRef(), C);
  if (!modOrErr) {
    std::cerr << "Failed to parse bitcode: " << path << "\n";
    logAllUnhandledErrors(modOrErr.takeError(), errs(), "");
    return nullptr;
  }

  if (*modOrErr) {
    (*modOrErr)->setDataLayout("e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128");
    (*modOrErr)->setTargetTriple("x86_64-unknown-linux-gnu");
  }

  return std::move(*modOrErr);
}

static bool loadProtobuf(const std::string& path, lifted_ast::Program& P) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  return P.ParseFromIstream(&in);
}

static bool saveProtobuf(const std::string& path, const lifted_ast::Program& P) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return false;
  return P.SerializeToOstream(&out);
}

int main(int argc, char** argv) {
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
  if (!loadProtobuf(args[1], P)) {
    std::cerr << "Failed to read protobuf: " << args[1] << "\n";
    return 1;
  }

  LLVMContext C;
  std::unique_ptr<Module> M = loadBitcodeModule(args[0], C);
  if (!M) return 1;

  StructType* StateTy = StructType::getTypeByName(C, "State");
  if (!StateTy) {
    std::cerr << "ERROR: Could not find identified struct type 'State' in module.\n";
    return 1;
  }

  std::map<std::string, std::string> bbLlvmMapping;
  std::map<std::string, std::string> instrLlvmMapping;

  for (const auto& sec : P.sections()) {
    for (const auto& ch : sec.children()) {
      if (!ch.has_function()) continue;
      const lifted_ast::Function& fnAst = ch.function();
      if (!fnAst.has_entry_label()) continue;

      std::string entry = fnAst.entry_label();
      std::string liftedName = entry + "_lifted";

      auto itSe = P.symbol_table().find(entry);
      if (itSe != P.symbol_table().end() &&
          itSe->second.has_lifted_ref() &&
          !itSe->second.lifted_ref().empty()) {
        liftedName = itSe->second.lifted_ref();
      }

      if (Function* F = M->getFunction(liftedName)) {
        Step11LowerCtx LC{C, *M, StateTy, F, nullptr, &P, &fnAst, bbLlvmMapping, instrLlvmMapping};
        lowerFunction(LC);
      } else {
        errs() << "Warning: lifted function not found in module: " << liftedName << "\n";
      }
    }
  }

  for (const auto& kv : bbLlvmMapping)
    (*P.mutable_bb_llvm_mapping())[kv.first] = kv.second;
  for (const auto& kv : instrLlvmMapping)
    (*P.mutable_instr_llvm_mapping())[kv.first] = kv.second;

  std::string originalDL = M->getDataLayoutStr();
  std::string originalTriple = M->getTargetTriple();

  if (verifyModule(*M, &errs()))
    errs() << "verifyModule FAILED after Step 11 lowering.\n";

  M->setDataLayout(originalDL);
  M->setTargetTriple(originalTriple);

  if (printMode) {
    std::string jsonStr;
    google::protobuf::util::JsonPrintOptions opts;
    opts.add_whitespace = true;
    if (!google::protobuf::util::MessageToJsonString(P, &jsonStr, opts).ok())
      return 1;
    std::ofstream out(args[3], std::ios::trunc);
    out << jsonStr;
  } else {
    if (!saveProtobuf(args[3], P)) return 1;
  }

  {
    std::error_code EC;
    raw_fd_ostream os(args[2], EC, sys::fs::OF_None);
    if (EC) return 1;
    if (printMode) M->print(os, nullptr);
    else WriteBitcodeToFile(*M, os);
    os.flush();
  }

  if (!printIrPath.empty()) {
    std::error_code EC;
    raw_fd_ostream os(printIrPath, EC, sys::fs::OF_None);
    if (!EC) {
      M->print(os, nullptr);
      os.flush();
    }
  }

  return 0;
}
