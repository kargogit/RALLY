// llvmLowerVectorSSE.cpp
// Step 14 – SSE vector lowering pass
// Supported opcodes: MOVDQA, MOVAPS, MOVUPS, ANDPS, PXOR
//
// Fully integrated with llvmLiftShared.hpp – all duplicated utilities,
// register decoding, state layout, lowering context, memory resolution,
// and erasure helpers are consumed from the shared header.

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

// ================================================================
// Step-specific helpers (not provided by the shared header)
// ================================================================

/// Collect every Instruction in \p BB whose !ast_instr_id equals \p instrId.
static std::vector<Instruction*> getAstIdRange(BasicBlock* BB,
                                                const std::string& instrId) {
  std::vector<Instruction*> range;
  if (instrId.empty())
    return range;
  for (auto& I : *BB)
    if (getAstInstrId(&I) == instrId)
      range.push_back(&I);
  return range;
}

/// Transitively erase every instruction in \p roots (and any users that
/// carry the same or no ast_instr_id), replacing lingering uses with undef.
/// Terminators are preserved.
static void eraseOldCodeAndUsers(const std::vector<Instruction*>& roots,
                                 const std::string& instrId) {
  std::set<Instruction*> toDelete;
  std::vector<Instruction*> worklist = roots;
  while (!worklist.empty()) {
    Instruction* I = worklist.back();
    worklist.pop_back();
    if (!I)
      continue;
    if (!toDelete.insert(I).second)
      continue;
    for (User* U : I->users()) {
      if (auto* UI = dyn_cast<Instruction>(U)) {
        std::string uid = getAstInstrId(UI);
        if (uid.empty() || uid == instrId)
          worklist.push_back(UI);
      }
    }
  }
  std::vector<Instruction*> finalDelete;
  for (auto* I : toDelete) {
    if (I->isTerminator())
      continue;
    finalDelete.push_back(I);
  }
  for (auto* I : finalDelete)
    if (!I->use_empty())
      I->replaceAllUsesWith(UndefValue::get(I->getType()));
  for (auto* I : finalDelete)
    I->eraseFromParent();
}

/// After new code has been emitted between \p prev and \p insertBefore,
/// stamp every new instruction with !ast_instr_id, pick a primary
/// result, name it, and record the mapping.
static void finalizeNewInstructions(FnLowerCtx& LC,
                                    BasicBlock* BB,
                                    Instruction* prev,
                                    Instruction* insertBefore,
                                    const std::string& instrId) {
  if (instrId.empty())
    return;
  Instruction* start = prev ? prev->getNextNode() : &BB->front();
  if (!start || start == insertBefore)
    return;

  MDNode* md = MDNode::get(LC.C, {MDString::get(LC.C, instrId)});
  Instruction* primary = nullptr;
  for (Instruction* I = start; I != insertBefore; I = I->getNextNode()) {
    I->setMetadata("ast_instr_id", md);
    if (!primary && !I->getType()->isVoidTy()) {
      if (isa<LoadInst>(I) || isa<CallInst>(I) || isa<AtomicRMWInst>(I) ||
          isa<AtomicCmpXchgInst>(I) || isa<FPMathOperator>(I))
        primary = I;
    }
  }
  if (!primary) {
    for (Instruction* I = start; I != insertBefore; I = I->getNextNode()) {
      if (!I->getType()->isVoidTy()) {
        primary = I;
        break;
      }
    }
  }
  if (primary && !primary->getType()->isVoidTy()) {
    primary->setName("instr_" + instrId + "_lowered");
    LC.instrLlvmMapping[instrId] = primary->getName().str();
  } else {
    LC.instrLlvmMapping.erase(instrId);
  }
}

// ================================================================
// SSE vector lowering logic
// ================================================================

static void lowerVectorSSE(FnLowerCtx& LC,
                           const lifted_ast::Instruction& insn,
                           IRBuilder<>& B,
                           const std::string& instrId) {
  if (insn.operands_size() < 2)
    return;

  const std::string opc = toUpper(insn.opcode());

  const bool isPS = (opc == "MOVAPS" || opc == "MOVUPS" || opc == "ANDPS");
  const bool isDQ = (opc == "MOVDQA" || opc == "PXOR");

  // View type – the LLVM type that matches the instruction's interpretation
  // of the 128-bit vector.  The State always stores <2 x double>; we bitcast
  // to/from the view type around every operation.
  Type* viewTy = nullptr;
  if (isPS)
    viewTy = FixedVectorType::get(B.getFloatTy(), 4);   // <4 x float>
  else if (isDQ)
    viewTy = FixedVectorType::get(B.getInt64Ty(), 2);   // <2 x i64>
  else
    return;

  Type* stateVecTy = FixedVectorType::get(B.getDoubleTy(), 2); // <2 x double>

  const auto& dstOp = insn.operands(0);
  const auto& srcOp = insn.operands(1);

  RegInfo dstRi = dstOp.has_register_() ? decodeReg(dstOp.register_()) : RegInfo{};
  RegInfo srcRi = srcOp.has_register_() ? decodeReg(srcOp.register_()) : RegInfo{};

  const bool dstIsReg = dstOp.has_register_() && dstRi.isValid && dstRi.isXmm;
  const bool srcIsReg = srcOp.has_register_() && srcRi.isValid && srcRi.isXmm;
  const bool dstIsMem = dstOp.has_memory();
  const bool srcIsMem = srcOp.has_memory();

  // MOVAPS / MOVDQA require 16-byte alignment; everything else is natural (1).
  const unsigned alignVal =
      (opc == "MOVAPS" || opc == "MOVDQA") ? 16 : 1;

  Value* resultVec = nullptr; // in stateVecTy, or nullptr on failure

  // ---- MOV family ----
  if (opc == "MOVDQA" || opc == "MOVAPS" || opc == "MOVUPS") {

    if (dstIsReg && srcIsReg) {
      // Reg ← Reg : direct state-level copy
      Value* srcVal = loadXmm(LC, B, srcRi.xmmIndex);
      storeXmm(LC, B, dstRi.xmmIndex, srcVal);
      resultVec = srcVal;
    } else if (dstIsReg && srcIsMem) {
      // Reg ← Mem : load view, bitcast to state, store state
      MemAddr ma = resolveMemAddress(LC, B, srcOp, 16, instrId);
      LoadInst* ld = B.CreateLoad(viewTy, ma.ptr, "vec.ld");
      ld->setAlignment(Align(alignVal));
      if (ma.isSymbolic && !ma.symName.empty())
        attachPicRelocations(ld, ma.symName, LC.C);
      attachAstInstrId(ld, instrId, LC.C);
      Value* stateVal = B.CreateBitCast(ld, stateVecTy, "vec.bc.state");
      storeXmm(LC, B, dstRi.xmmIndex, stateVal);
      resultVec = stateVal;
    } else if (dstIsMem && srcIsReg) {
      // Mem ← Reg : load state, bitcast to view, store view
      Value* srcVal = loadXmm(LC, B, srcRi.xmmIndex);
      Value* viewVal = B.CreateBitCast(srcVal, viewTy, "vec.bc.view");
      MemAddr ma = resolveMemAddress(LC, B, dstOp, 16, instrId);
      StoreInst* st = B.CreateStore(viewVal, ma.ptr);
      st->setAlignment(Align(alignVal));
      if (ma.isSymbolic && !ma.symName.empty())
        attachPicRelocations(st, ma.symName, LC.C);
      attachAstInstrId(st, instrId, LC.C);
      resultVec = srcVal;
    }

  // ---- bitwise family ----
  } else if (opc == "ANDPS" || opc == "PXOR") {
    if (dstIsReg) {
      Value* dstState = loadXmm(LC, B, dstRi.xmmIndex);
      Value* dstView  = B.CreateBitCast(dstState, viewTy, "op.dst.bc");

      Value* srcView = nullptr;
      if (srcIsReg) {
        Value* srcState = loadXmm(LC, B, srcRi.xmmIndex);
        srcView = B.CreateBitCast(srcState, viewTy, "op.src.bc");
      } else if (srcIsMem) {
        MemAddr ma = resolveMemAddress(LC, B, srcOp, 16, instrId);
        LoadInst* ld = B.CreateLoad(viewTy, ma.ptr, "op.src.ld");
        ld->setAlignment(Align(1));
        if (ma.isSymbolic && !ma.symName.empty())
          attachPicRelocations(ld, ma.symName, LC.C);
        attachAstInstrId(ld, instrId, LC.C);
        srcView = ld;
      }

      if (srcView) {
        Value* resView = (opc == "ANDPS")
                             ? B.CreateAnd(dstView, srcView, "andps.res")
                             : B.CreateXor(dstView, srcView, "pxor.res");
        Value* resState = B.CreateBitCast(resView, stateVecTy, "op.res.bc");
        storeXmm(LC, B, dstRi.xmmIndex, resState);
        resultVec = resState;
      }
    }
  }

  // Record the primary result in the instruction mapping.
  if (resultVec && !resultVec->getType()->isVoidTy()) {
    if (auto* I = dyn_cast<Instruction>(resultVec)) {
      I->setName("instr_" + instrId + "_vec");
      attachAstInstrId(I, instrId, LC.C);
      LC.instrLlvmMapping[instrId] = I->getName().str();
    }
  }
}

// ================================================================
// Per-function processing
// ================================================================

static void processFunction(FnLowerCtx& LC) {
  if (!LC.F || LC.F->empty())
    return;

  // Build a quick id → AST-instruction lookup.
  std::map<std::string, const lifted_ast::Instruction*> idToAstInsn;
  for (const auto& bbAst : LC.FnAst->basic_blocks())
    for (const auto& ie : bbAst.instructions())
      if (ie.instruction().has_id())
        idToAstInsn[ie.instruction().id()] = &ie.instruction();

  for (auto& BB : *LC.F) {
    // Gather placeholder calls in this block.
    std::vector<CallInst*> placeholders;
    for (auto& I : BB)
      if (auto* CI = dyn_cast<CallInst>(&I))
        if (getPlaceholderOpcode(CI))
          placeholders.push_back(CI);

    for (auto* CI : placeholders) {
      auto opcOpt = getPlaceholderOpcode(CI);
      if (!opcOpt)
        continue;

      std::string opcUpper = toUpper(*opcOpt);
      if (opcUpper != "MOVDQA" && opcUpper != "MOVAPS" &&
          opcUpper != "MOVUPS" && opcUpper != "ANDPS" && opcUpper != "PXOR")
        continue;

      std::string instrId = getAstInstrId(CI);
      auto astIt = idToAstInsn.find(instrId);
      if (astIt == idToAstInsn.end())
        continue;

      // Collect old instructions with this ID, but exclude the placeholder
      // itself – we erase it separately at the very end.
      std::vector<Instruction*> oldInstrs = getAstIdRange(&BB, instrId);
      oldInstrs.erase(
          std::remove(oldInstrs.begin(), oldInstrs.end(), CI),
          oldInstrs.end());

      Instruction* insertBefore =
          oldInstrs.empty() ? CI : oldInstrs.front();
      Instruction* prev = insertBefore->getPrevNode();

      IRBuilder<> B(insertBefore);
      LC.stateGepCache.clear();

      lowerVectorSSE(LC, *astIt->second, B, instrId);
      finalizeNewInstructions(LC, &BB, prev, insertBefore, instrId);
      eraseOldCodeAndUsers(oldInstrs, instrId);
      eraseDummyStoresAfter(CI->getNextNode(), instrId);
      CI->eraseFromParent();
    }
  }
}

// ================================================================
// Module driver
// ================================================================

static std::unique_ptr<Module> loadBitcodeModule(const std::string& path,
                                                  LLVMContext& C) {
  auto bufOrErr = MemoryBuffer::getFile(path);
  if (!bufOrErr)
    return nullptr;
  auto modOrErr = parseBitcodeFile(bufOrErr->get()->getMemBufferRef(), C);
  if (!modOrErr) {
    logAllUnhandledErrors(modOrErr.takeError(), errs(), "");
    return nullptr;
  }
  return std::move(*modOrErr);
}

static bool loadProtobuf(const std::string& path, lifted_ast::Program& P) {
  std::ifstream in(path, std::ios::binary);
  return in && P.ParseFromIstream(&in);
}

static bool saveProtobuf(const std::string& path, const lifted_ast::Program& P) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  return out && P.SerializeToOstream(&out);
}

int main(int argc, char** argv) {
  bool printMode = false;
  std::vector<std::string> args;
  std::string printIrPath;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--print-ir" && i + 1 < argc)
      printIrPath = argv[++i];
    else if (a == "--print")
      printMode = true;
    else
      args.push_back(a);
  }
  if (args.size() < 4)
    return 1;

  lifted_ast::Program P;
  if (!loadProtobuf(args[1], P))
    return 1;

  LLVMContext C;
  auto M = loadBitcodeModule(args[0], C);
  if (!M)
    return 1;

  StructType* StateTy = StructType::getTypeByName(C, "State");
  if (!StateTy)
    return 1;

  std::map<std::string, std::string> instrLlvmMapping;
  for (const auto& kv : P.instr_llvm_mapping())
    instrLlvmMapping[kv.first] = kv.second;

  for (const auto& sec : P.sections()) {
    for (const auto& ch : sec.children()) {
      if (!ch.has_function())
        continue;
      const lifted_ast::Function& fnAst = ch.function();
      if (!fnAst.has_entry_label())
        continue;

      std::string entry = fnAst.entry_label();
      std::string liftedName = entry + "_lifted";
      auto itSe = P.symbol_table().find(entry);
      if (itSe != P.symbol_table().end() && itSe->second.has_lifted_ref() &&
          !itSe->second.lifted_ref().empty())
        liftedName = itSe->second.lifted_ref();

      if (Function* F = M->getFunction(liftedName)) {
        // The first argument of a lifted function is always %State*.
        Value* stateArg = F->arg_empty() ? nullptr : &*F->arg_begin();
        FnLowerCtx LC(C, *M, StateTy, F, stateArg, &P, &fnAst,
                       instrLlvmMapping);

        for (auto& BB : *F)
          LC.bbIdToLlvm[BB.getName().str()] = &BB;

        processFunction(LC);
      }
    }
  }

  // Write back the updated mapping.
  for (const auto& kv : instrLlvmMapping)
    (*P.mutable_instr_llvm_mapping())[kv.first] = kv.second;

  verifyModule(*M, &errs());

  // ---- outputs ----
  if (printMode) {
    std::string jsonStr;
    google::protobuf::util::JsonPrintOptions opts;
    opts.add_whitespace = true;
    if (google::protobuf::util::MessageToJsonString(P, &jsonStr, opts).ok())
      std::ofstream(args[3], std::ios::trunc) << jsonStr;
  } else {
    saveProtobuf(args[3], P);
  }

  {
    std::error_code EC;
    raw_fd_ostream os(args[2], EC, sys::fs::OF_None);
    if (!EC) {
      if (printMode)
        M->print(os, nullptr);
      else
        WriteBitcodeToFile(*M, os);
      os.flush();
    }
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
