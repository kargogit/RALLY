// llvmLowerFpAtomic.cpp (Implementation of Step 12)
//
// Supported Instructions:
// Floating-Point Data Movement: MOVSS, MOVSD
// Floating-Point Arithmetic: ADDSS, ADDSD, MULSS
// Floating-Point Conversion: CVTTSS2SI
// Floating-Point Comparison: UCOMISS, COMISS
// Atomic Operations: XCHG (reg <-> mem), INC (LOCK memory)

#include "llvmLiftShared.hpp"

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <google/protobuf/util/json_util.h>

using namespace llvm;
using namespace llvm_lift;

// ----------------------------- Flag computation -----------------------------

static Value* computePF(IRBuilder<> &B, Value *resIntN) {
  Value *lo8 = B.CreateTrunc(resIntN, B.getInt8Ty(), "pf.lo8");
  Function *ctpop = Intrinsic::getDeclaration(
      B.GetInsertBlock()->getModule(), Intrinsic::ctpop, {B.getInt8Ty()});
  Value *pop = B.CreateCall(ctpop, {lo8}, "pf.pop");
  Value *lsb = B.CreateAnd(pop, B.getInt8(1), "pf.lsb");
  Value *odd = B.CreateICmpEQ(lsb, B.getInt8(1), "pf.odd");
  return B.CreateNot(odd, "pf");
}

// ----------------------------- Erasure & Metadata Utilities -----------------------------

static std::vector<Instruction*> getAstIdRange(BasicBlock *BB, const std::string &instrId) {
  std::vector<Instruction*> range;
  if (instrId.empty()) return range;

  for (auto &I : *BB) {
    if (getAstInstrId(&I) == instrId) {
      range.push_back(&I);
    }
  }
  return range;
}

static void eraseOldCodeAndUsers(const std::vector<Instruction*> &roots,
                                 const std::string &instrId) {
  std::set<Instruction*> to_delete;
  std::vector<Instruction*> worklist = roots;

  while (!worklist.empty()) {
    Instruction *I = worklist.back();
    worklist.pop_back();

    if (!I) continue;
    if (!to_delete.insert(I).second) continue;

    for (User *U : I->users()) {
      if (auto *UI = dyn_cast<Instruction>(U)) {
        std::string uid = getAstInstrId(UI);
        if (uid.empty() || uid == instrId) {
          worklist.push_back(UI);
        }
      }
    }
  }

  std::vector<Instruction*> final_delete;
  for (auto *I : to_delete) {
    if (I->isTerminator()) continue;
    final_delete.push_back(I);
  }

  for (auto *I : final_delete) {
    if (!I->use_empty()) I->replaceAllUsesWith(UndefValue::get(I->getType()));
  }

  for (auto *I : final_delete) I->eraseFromParent();
}

static void finalizeNewInstructions(FnLowerCtx &LC, BasicBlock *BB,
                                    Instruction *prev, Instruction *insertBefore,
                                    const std::string &instrId) {
  if (instrId.empty()) return;
  Instruction *start = prev ? prev->getNextNode() : &BB->front();
  if (!start || start == insertBefore) return;

  MDNode *md = MDNode::get(LC.C, {MDString::get(LC.C, instrId)});
  Instruction *primary = nullptr;

  for (Instruction *I = start; I != insertBefore; I = I->getNextNode()) {
    I->setMetadata("ast_instr_id", md);
    if (!primary && !I->getType()->isVoidTy()) {
      if (isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I) ||
          isa<FPMathOperator>(I) || isa<CallInst>(I) || isa<LoadInst>(I)) {
        primary = I;
      }
    }
  }

  if (!primary) {
    for (Instruction *I = start; I != insertBefore; I = I->getNextNode()) {
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

// ----------------------------- FP Lowering Functions -----------------------------

static void lowerMOVSS_load(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn, const std::string& astInstrId) {
  if (insn.operands_size() < 2) return;
  RegInfo dri = decodeReg(insn.operands(0).register_());
  Value *ld = loadFromMem(LC, B, insn.operands(1), B.getFloatTy(), astInstrId);

  auto *f4Ty = FixedVectorType::get(B.getFloatTy(), 4);
  Value *f4Vec = B.CreateInsertElement(ConstantAggregateZero::get(f4Ty), ld, (uint64_t)0, "movss.ins");
  Value *d2Vec = B.CreateBitCast(f4Vec, FixedVectorType::get(B.getDoubleTy(), 2), "movss.bc");
  storeXmm(LC, B, dri.xmmIndex, d2Vec);
}

static void lowerMOVSS_store(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn, const std::string& astInstrId) {
  if (insn.operands_size() < 2) return;
  RegInfo sri = decodeReg(insn.operands(1).register_());

  Value *d2Vec = loadXmm(LC, B, sri.xmmIndex);
  Value *f4Vec = B.CreateBitCast(d2Vec, FixedVectorType::get(B.getFloatTy(), 4), "movss.bc");
  Value *f32Val = B.CreateExtractElement(f4Vec, (uint64_t)0, "movss.ext");

  storeToMem(LC, B, insn.operands(0), f32Val, B.getFloatTy(), astInstrId);
}

static void lowerMULSS(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn, const std::string& astInstrId) {
  if (insn.operands_size() < 2) return;
  RegInfo dri = decodeReg(insn.operands(0).register_());

  auto *f4Ty = FixedVectorType::get(B.getFloatTy(), 4);
  Value *f4VecDst = B.CreateBitCast(loadXmm(LC, B, dri.xmmIndex), f4Ty, "mulss.dst.bc");
  Value *f0Dst = B.CreateExtractElement(f4VecDst, (uint64_t)0, "mulss.dst.f0");
  Value *f0Src = nullptr;

  if (insn.operands(1).has_register_()) {
    Value *f4VecSrc = B.CreateBitCast(loadXmm(LC, B, decodeReg(insn.operands(1).register_()).xmmIndex), f4Ty);
    f0Src = B.CreateExtractElement(f4VecSrc, (uint64_t)0, "mulss.src.f0");
  } else if (insn.operands(1).has_memory()) {
    f0Src = loadFromMem(LC, B, insn.operands(1), B.getFloatTy(), astInstrId);
  }

  Value *prod = B.CreateFMul(f0Dst, f0Src ? f0Src : f0Dst, "mulss.prod");
  Value *f4VecNew = B.CreateInsertElement(f4VecDst, prod, (uint64_t)0, "mulss.ins");
  storeXmm(LC, B, dri.xmmIndex, B.CreateBitCast(f4VecNew, FixedVectorType::get(B.getDoubleTy(), 2)));
}

static void lowerADDSS(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn, const std::string& astInstrId) {
  if (insn.operands_size() < 2) return;
  RegInfo dri = decodeReg(insn.operands(0).register_());

  auto *f4Ty = FixedVectorType::get(B.getFloatTy(), 4);
  Value *f4VecDst = B.CreateBitCast(loadXmm(LC, B, dri.xmmIndex), f4Ty, "addss.dst.bc");
  Value *f0Dst = B.CreateExtractElement(f4VecDst, (uint64_t)0, "addss.dst.f0");
  Value *f0Src = nullptr;

  if (insn.operands(1).has_register_()) {
    Value *f4VecSrc = B.CreateBitCast(loadXmm(LC, B, decodeReg(insn.operands(1).register_()).xmmIndex), f4Ty);
    f0Src = B.CreateExtractElement(f4VecSrc, (uint64_t)0, "addss.src.f0");
  } else if (insn.operands(1).has_memory()) {
    f0Src = loadFromMem(LC, B, insn.operands(1), B.getFloatTy(), astInstrId);
  }

  Value *sum = B.CreateFAdd(f0Dst, f0Src ? f0Src : f0Dst, "addss.sum");
  Value *f4VecNew = B.CreateInsertElement(f4VecDst, sum, (uint64_t)0, "addss.ins");
  storeXmm(LC, B, dri.xmmIndex, B.CreateBitCast(f4VecNew, FixedVectorType::get(B.getDoubleTy(), 2)));
}

static void lowerCVTTSS2SI(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn, const std::string& astInstrId) {
  if (insn.operands_size() < 2) return;
  RegInfo dri = decodeReg(insn.operands(0).register_());
  Value *f32Val = nullptr;

  if (insn.operands(1).has_register_()) {
    Value *f4Vec = B.CreateBitCast(loadXmm(LC, B, decodeReg(insn.operands(1).register_()).xmmIndex), FixedVectorType::get(B.getFloatTy(), 4));
    f32Val = B.CreateExtractElement(f4Vec, (uint64_t)0, "cvttss.f0");
  } else if (insn.operands(1).has_memory()) {
    f32Val = loadFromMem(LC, B, insn.operands(1), B.getFloatTy(), astInstrId);
  }

  if (f32Val) writeGprSubreg(LC, B, dri, B.CreateFPToSI(f32Val, intTy(LC.C, dri.bitWidth), "cvttss.tosi"));
}

static void lowerMOVSD_load(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn, const std::string& astInstrId) {
  if (insn.operands_size() < 2) return;
  RegInfo dri = decodeReg(insn.operands(0).register_());

  Value *ld = loadFromMem(LC, B, insn.operands(1), B.getDoubleTy(), astInstrId);
  storeXmm(LC, B, dri.xmmIndex, B.CreateInsertElement(loadXmm(LC, B, dri.xmmIndex), ld, (uint64_t)0, "movsd.ins"));
}

static void lowerADDSD(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn, const std::string& astInstrId) {
  if (insn.operands_size() < 2) return;
  RegInfo dri = decodeReg(insn.operands(0).register_());

  Value *d2VecDst = loadXmm(LC, B, dri.xmmIndex);
  Value *f0Dst = B.CreateExtractElement(d2VecDst, (uint64_t)0, "addsd.dst.f0");
  Value *f0Src = nullptr;

  if (insn.operands(1).has_register_()) {
    f0Src = B.CreateExtractElement(loadXmm(LC, B, decodeReg(insn.operands(1).register_()).xmmIndex), (uint64_t)0, "addsd.src.f0");
  } else if (insn.operands(1).has_memory()) {
    f0Src = loadFromMem(LC, B, insn.operands(1), B.getDoubleTy(), astInstrId);
  }

  Value *sum = B.CreateFAdd(f0Dst, f0Src ? f0Src : f0Dst, "addsd.sum");
  storeXmm(LC, B, dri.xmmIndex, B.CreateInsertElement(d2VecDst, sum, (uint64_t)0, "addsd.ins"));
}

static void lowerUCOMISS(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Instruction &insn, bool ordered, const std::string& astInstrId) {
  if (insn.operands_size() < 2) return;
  auto *f4Ty = FixedVectorType::get(B.getFloatTy(), 4);
  Value *f0a = nullptr, *f0b = nullptr;

  if (insn.operands(0).has_register_()) {
    f0a = B.CreateExtractElement(B.CreateBitCast(loadXmm(LC, B, decodeReg(insn.operands(0).register_()).xmmIndex), f4Ty), (uint64_t)0);
  }

  if (insn.operands(1).has_register_()) {
    f0b = B.CreateExtractElement(B.CreateBitCast(loadXmm(LC, B, decodeReg(insn.operands(1).register_()).xmmIndex), f4Ty), (uint64_t)0);
  } else if (insn.operands(1).has_memory()) {
    f0b = loadFromMem(LC, B, insn.operands(1), B.getFloatTy(), astInstrId);
  }

  if (!f0a || !f0b) return;

  Value *unord = B.CreateFCmpUNO(f0a, f0b, "ucomiss.unord");
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

static void lowerAtomicXCHG(FnLowerCtx &LC, const lifted_ast::Instruction &insn, BasicBlock *BB) {
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
  std::vector<Instruction*> oldInstrs = getAstIdRange(BB, instrId);
  Instruction *insertBefore = oldInstrs.empty() ? nullptr : oldInstrs.front();
  if (!insertBefore) return;

  Instruction *prev = insertBefore->getPrevNode();
  IRBuilder<> B(insertBefore);
  LC.stateGepCache.clear();

  unsigned accessSize = rri.bitWidth / 8;
  MemAddr ma = resolveMemAddress(LC, B, *memOp, accessSize, instrId);

  AtomicRMWInst *rmw = B.CreateAtomicRMW(
      AtomicRMWInst::Xchg, ma.ptr, truncOrZext(B, readGprSubreg(LC, B, rri), intTy(LC.C, rri.bitWidth)),
      MaybeAlign(std::max<unsigned>(1, accessSize)), AtomicOrdering::SequentiallyConsistent);

  if (ma.align.value() > 1) rmw->setAlignment(ma.align);
  if (ma.isSymbolic) attachPicRelocations(rmw, ma.symName, LC.C);
  attachAstInstrId(rmw, instrId, LC.C);

  writeGprSubreg(LC, B, rri, rmw);

  finalizeNewInstructions(LC, BB, prev, insertBefore, instrId);
  eraseOldCodeAndUsers(oldInstrs, instrId);
}

static void lowerAtomicINC(FnLowerCtx &LC, const lifted_ast::Instruction &insn, BasicBlock *BB) {
  if (insn.operands_size() < 1 || !insn.operands(0).has_memory()) return;
  std::string instrId = insn.has_id() ? insn.id() : "";
  std::vector<Instruction*> oldInstrs = getAstIdRange(BB, instrId);
  Instruction *insertBefore = oldInstrs.empty() ? nullptr : oldInstrs.front();
  if (!insertBefore) return;

  Instruction *prev = insertBefore->getPrevNode();
  IRBuilder<> B(insertBefore);
  LC.stateGepCache.clear();

  unsigned accessSize = std::max((unsigned)1, memSizeBytesFromOperand(insn.operands(0)));
  MemAddr ma = resolveMemAddress(LC, B, insn.operands(0), accessSize, instrId);
  Type *opTy = intTy(LC.C, accessSize * 8);
  Value *one = ConstantInt::get(opTy, 1);

  AtomicRMWInst *oldVal = B.CreateAtomicRMW(
      AtomicRMWInst::Add, ma.ptr, one,
      MaybeAlign(accessSize), AtomicOrdering::SequentiallyConsistent);

  if (ma.align.value() > 1) oldVal->setAlignment(ma.align);
  if (ma.isSymbolic) attachPicRelocations(oldVal, ma.symName, LC.C);
  attachAstInstrId(oldVal, instrId, LC.C);

  Value *newVal = B.CreateAdd(oldVal, one, "atomic.inc.new");
  storeFlag(LC, B, ZF, B.CreateICmpEQ(newVal, ConstantInt::get(opTy, 0), "zf"));
  storeFlag(LC, B, SF, B.CreateICmpSLT(newVal, ConstantInt::get(opTy, 0), "sf"));
  storeFlag(LC, B, PF, computePF(B, newVal));
  storeFlag(LC, B, OF, B.CreateExtractValue(B.CreateCall(Intrinsic::getDeclaration(&LC.M, Intrinsic::sadd_with_overflow, {opTy}), {oldVal, one}), 1, "of"));
  storeFlag(LC, B, AF, B.CreateICmpNE(B.CreateAnd(B.CreateXor(B.CreateXor(oldVal, one), newVal), ConstantInt::get(opTy, 0x10)), ConstantInt::get(opTy, 0)));

  // INC does not modify CF - preserve previous value locally
  Value *cfPtr = getStateFieldPtr(LC, B, kFlagsBase + (unsigned)CF);
  Value *prevCf = B.CreateLoad(B.getInt1Ty(), cfPtr, "prev.cf");
  B.CreateStore(prevCf, cfPtr);

  finalizeNewInstructions(LC, BB, prev, insertBefore, instrId);
  eraseOldCodeAndUsers(oldInstrs, instrId);
}

static void lowerAtomicXADD(FnLowerCtx &LC, const lifted_ast::Instruction &insn, BasicBlock *BB) {
  if (insn.operands_size() < 2) return;

  const lifted_ast::Operand *memOp = nullptr, *regOp = nullptr;
  if (insn.operands(0).has_memory() && insn.operands(1).has_register_()) {
    memOp = &insn.operands(0); regOp = &insn.operands(1);
  } else if (insn.operands(1).has_memory() && insn.operands(0).has_register_()) {
    memOp = &insn.operands(1); regOp = &insn.operands(0);
  }
  if (!memOp || !regOp) return;

  std::string instrId = insn.has_id() ? insn.id() : "";
  std::vector<Instruction*> oldInstrs = getAstIdRange(BB, instrId);
  Instruction *insertBefore = oldInstrs.empty() ? nullptr : oldInstrs.front();
  if (!insertBefore) return;

  Instruction *prev = insertBefore->getPrevNode();
  IRBuilder<> B(insertBefore);
  LC.stateGepCache.clear();

  RegInfo rri = decodeReg(regOp->register_());
  unsigned accessSize = std::max((unsigned)1, memSizeBytesFromOperand(*memOp));
  MemAddr ma = resolveMemAddress(LC, B, *memOp, accessSize, instrId);
  Type *opTy = intTy(LC.C, accessSize * 8);

  Value *srcVal = truncOrZext(B, readGprSubreg(LC, B, rri), opTy);

  AtomicRMWInst *oldVal = B.CreateAtomicRMW(
      AtomicRMWInst::Add, ma.ptr, srcVal,
      MaybeAlign(accessSize), AtomicOrdering::SequentiallyConsistent);

  if (ma.align.value() > 1) oldVal->setAlignment(ma.align);
  if (ma.isSymbolic) attachPicRelocations(oldVal, ma.symName, LC.C);
  attachAstInstrId(oldVal, instrId, LC.C);

  // XADD: register receives the old memory value
  writeGprSubreg(LC, B, rri, oldVal);

  Value *newVal = B.CreateAdd(oldVal, srcVal, "atomic.xadd.new");

  // Arithmetic flags (exactly as a normal ADD/XADD would set them)
  storeFlag(LC, B, CF, B.CreateExtractValue(
      B.CreateCall(Intrinsic::getDeclaration(&LC.M, Intrinsic::uadd_with_overflow, {opTy}), {oldVal, srcVal}), 1, "cf"));
  storeFlag(LC, B, OF, B.CreateExtractValue(
      B.CreateCall(Intrinsic::getDeclaration(&LC.M, Intrinsic::sadd_with_overflow, {opTy}), {oldVal, srcVal}), 1, "of"));
  storeFlag(LC, B, ZF, B.CreateICmpEQ(newVal, ConstantInt::get(opTy, 0), "zf"));
  storeFlag(LC, B, SF, B.CreateICmpSLT(newVal, ConstantInt::get(opTy, 0), "sf"));
  storeFlag(LC, B, PF, computePF(B, newVal));
  storeFlag(LC, B, AF, B.CreateICmpNE(
      B.CreateAnd(B.CreateXor(B.CreateXor(oldVal, srcVal), newVal), ConstantInt::get(opTy, 0x10)),
      ConstantInt::get(opTy, 0)));

  finalizeNewInstructions(LC, BB, prev, insertBefore, instrId);
  eraseOldCodeAndUsers(oldInstrs, instrId);
}

// ----------------------------- Main Processing -----------------------------

static void processFunction(FnLowerCtx &LC) {
  if (!LC.F || LC.F->empty()) return;

  auto instIt = LC.F->getEntryBlock().begin();
  for (const auto &ss : LC.FnAst->stack_slots()) {
    while (instIt != LC.F->getEntryBlock().end() && !isa<AllocaInst>(*instIt)) {
      ++instIt;
    }
    if (instIt == LC.F->getEntryBlock().end()) break;

    StackSlotInfo info;
    info.name = ss.has_name() ? ss.name() : "slot";
    info.baseReg = ss.has_register_() ? ss.register_() : "RBP";
    info.startOff = ss.has_offset() ? ss.offset() : 0;
    info.size = ss.has_size() ? ss.size() : 8;
    info.align = ss.has_alignment() ? ss.alignment() : 1;
    info.alloca = cast<AllocaInst>(&*instIt);

    LC.promotedSlots.push_back(info);
    ++instIt;
  }

  std::map<std::string, const lifted_ast::Instruction*> idToAstInsn;
  for (const auto &bbAst : LC.FnAst->basic_blocks()) {
    for (const auto &ie : bbAst.instructions()) {
      if (ie.instruction().has_id()) idToAstInsn[ie.instruction().id()] = &ie.instruction();
    }
  }

  for (auto &BB : *LC.F) LC.bbIdToLlvm[BB.getName().str()] = &BB;

  // Pass 1: Handle structured atomic instructions
  for (const auto &bbAst : LC.FnAst->basic_blocks()) {
    std::string bbName = bbAst.has_start_label() && !bbAst.start_label().empty() ? bbAst.start_label() : (bbAst.has_id() && !bbAst.id().empty() ? bbAst.id() : "");
    BasicBlock *BB = nullptr;
    if (auto it = LC.bbIdToLlvm.find(bbName); it != LC.bbIdToLlvm.end()) BB = it->second;
    if (!BB) continue;

    for (const auto &ie : bbAst.instructions()) {
      const lifted_ast::Instruction &insn = ie.instruction();
      std::string opc = insn.has_opcode() ? toUpper(insn.opcode()) : "";

      if (isXchgWithMemory(insn)) { lowerAtomicXCHG(LC, insn, BB); continue; }
      if (hasLockPrefix(insn) && opc == "INC") lowerAtomicINC(LC, insn, BB);
      if (hasLockPrefix(insn) && opc == "XADD") lowerAtomicXADD(LC, insn, BB);
    }
  }

  // Pass 2: Find and replace FP placeholders
  for (auto &BB : *LC.F) {
    std::vector<CallInst*> placeholders;
    for (auto &I : BB) {
      if (auto *CI = dyn_cast<CallInst>(&I)) {
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

      std::vector<Instruction*> oldInstrs = getAstIdRange(&BB, instrId);
      Instruction *insertBefore = oldInstrs.empty() ? CI : oldInstrs.front();
      Instruction *prev = insertBefore->getPrevNode();

      Instruction *afterOldCode = nullptr;
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
        if (insn.operands(0).has_register_() && decodeReg(insn.operands(0).register_()).isXmm && insn.operands(1).has_memory()) { lowerMOVSS_load(LC, B, insn, instrId); handled = true; }
        else if (insn.operands(0).has_memory() && insn.operands(1).has_register_() && decodeReg(insn.operands(1).register_()).isXmm) { lowerMOVSS_store(LC, B, insn, instrId); handled = true; }
      } else if (opc == "ADDSS") { lowerADDSS(LC, B, insn, instrId); handled = true; }
      else if (opc == "MULSS") { lowerMULSS(LC, B, insn, instrId); handled = true; }
      else if (opc == "CVTTSS2SI") { lowerCVTTSS2SI(LC, B, insn, instrId); handled = true; }
      else if (opc == "MOVSD" && insn.operands_size() >= 2 && insn.operands(0).has_register_() && decodeReg(insn.operands(0).register_()).isXmm && insn.operands(1).has_memory()) { lowerMOVSD_load(LC, B, insn, instrId); handled = true; }
      else if (opc == "ADDSD") { lowerADDSD(LC, B, insn, instrId); handled = true; }
      else if (opc == "UCOMISS" || opc == "COMISS") { lowerUCOMISS(LC, B, insn, opc == "COMISS", instrId); handled = true; }

      if (handled) {
        finalizeNewInstructions(LC, &BB, prev, insertBefore, instrId);
        eraseOldCodeAndUsers(oldInstrs, instrId);

        // Erase orphaned dummy stores from Step 11 leftovers securely
        eraseDummyStoresAfter(afterOldCode, instrId);
      }
    }
  }
}

// ----------------------------- Module driver -----------------------------

static std::unique_ptr<Module> loadBitcodeModule(const std::string &path, LLVMContext &C) {
  auto bufOrErr = MemoryBuffer::getFile(path);
  if (!bufOrErr) return nullptr;
  auto modOrErr = parseBitcodeFile(bufOrErr->get()->getMemBufferRef(), C);
  if (!modOrErr) {
    logAllUnhandledErrors(modOrErr.takeError(), errs(), "");
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

  LLVMContext C;
  std::unique_ptr<Module> M = loadBitcodeModule(args[0], C);
  if (!M) return 1;

  StructType *StateTy = StructType::getTypeByName(C, "State");
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

      if (Function *F = M->getFunction(liftedName)) {
        FnLowerCtx LC{C, *M, StateTy, F, F->getArg(0), &P, &fnAst, instrLlvmMapping, nullptr};
        processFunction(LC);
      }
    }
  }

  for (const auto &kv : instrLlvmMapping) (*P.mutable_instr_llvm_mapping())[kv.first] = kv.second;
  verifyModule(*M, &errs());

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
    raw_fd_ostream os(args[2], EC, sys::fs::OF_None);
    if (!EC) { printMode ? M->print(os, nullptr) : WriteBitcodeToFile(*M, os); os.flush(); }
  }

  if (!printIrPath.empty()) {
    std::error_code EC;
    raw_fd_ostream os(printIrPath, EC, sys::fs::OF_None);
    if (!EC) { M->print(os, nullptr); os.flush(); }
  }

  return 0;
}
