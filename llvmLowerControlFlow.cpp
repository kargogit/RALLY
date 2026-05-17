// llvmLowerControlFlow.cpp
// =============================================================================
// LLVM CONTROL FLOW LOWERING (Step 13)
// Unified Edition using llvmLiftShared.hpp
//
// This version cleanly leverages the perfect common core shared library:
// - All state accessors, register decoding, sub-register ops
// - Universal RValue resolution & memory/symbol/GOT handling
// - ABI call emission (internal lifted + external System V ABI)
// - Placeholder detection & dummy-store erasure
// - Metadata attachment (AST IDs + PIC relocations where applicable)
// - Improved FnLowerCtx with full AST context
//
// Supported Instructions:
// Control Flow
// - CALL
// - JMP (direct + indirect + tail-call variants)
// - RET
// - IRET
// - SYSRET
// - SYSCALL
// - HLT
// Conditional Jumps (Jcc family)
// - JE/JZ, JNE/JNZ
// - JS, JNS
// - JO, JNO
// - JP/JPE, JNP/JPO
// - JB/JC/JNAE, JAE/JNB/JNC
// - JBE/JNA, JA/JNBE
// - JL/JNGE, JGE/JNL
// - JLE/JNG, JG/JNLE
// Loop Instructions
// - LOOP, LOOPE, LOOPNE
// =============================================================================
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include "ast.pb.h"
#include <google/protobuf/util/json_util.h>
#include "llvmLiftShared.hpp"

using namespace llvm;
using namespace llvm_lift;

// ----------------------------- Control-Flow-Specific Helpers -----------------------------
static void clobberFlags(FnLowerCtx &LC, IRBuilder<> &B) {
    Value *undefFlag = UndefValue::get(B.getInt1Ty());
    for (unsigned i = 0; i < kFlagCount; ++i) {
        storeFlag(LC, B, static_cast<FlagIndex>(i), undefFlag);
    }
}

static void lowerCall(FnLowerCtx &LC, CallInst *placeholder, const lifted_ast::Instruction &astInsn) {
    IRBuilder<> B(placeholder);
    LC.stateGepCache.clear();
    Instruction* nextInstr = placeholder->getNextNode();
    std::string instrId = getAstInstrId(placeholder);
    if (astInsn.operands_size() > 0) {
        emitFunctionCall(LC, B, astInsn.operands(0), false);
    }
    clobberFlags(LC, B);
    placeholder->eraseFromParent();
    eraseDummyStoresAfter(nextInstr, instrId);
}

static void lowerSyscall(FnLowerCtx &LC, CallInst *placeholder, const lifted_ast::Instruction &astInsn) {
    IRBuilder<> B(placeholder);
    LC.stateGepCache.clear();
    Instruction* nextInstr = placeholder->getNextNode();
    std::string instrId = getAstInstrId(placeholder);
    Function *syscallF = LC.M.getFunction("syscall");
    if (!syscallF) {
        FunctionType *sft = FunctionType::get(B.getInt64Ty(), {B.getInt64Ty()}, true);
        syscallF = Function::Create(sft, GlobalValue::ExternalLinkage, "syscall", &LC.M);
    }
    SmallVector<Value*, 7> args;
    args.push_back(loadGpr64(LC, B, gprFieldIndex64("RAX"))); // sysno
    args.push_back(loadGpr64(LC, B, gprFieldIndex64("RDI")));
    args.push_back(loadGpr64(LC, B, gprFieldIndex64("RSI")));
    args.push_back(loadGpr64(LC, B, gprFieldIndex64("RDX")));
    args.push_back(loadGpr64(LC, B, gprFieldIndex64("R10")));
    args.push_back(loadGpr64(LC, B, gprFieldIndex64("R8")));
    args.push_back(loadGpr64(LC, B, gprFieldIndex64("R9")));
    CallInst *ci = B.CreateCall(syscallF, args);
    storeGpr64(LC, B, gprFieldIndex64("RAX"), ci);
    clobberFlags(LC, B);
    placeholder->eraseFromParent();
    eraseDummyStoresAfter(nextInstr, instrId);
}

static void lowerTerminator(FnLowerCtx &LC, BasicBlock *BB, const lifted_ast::BasicBlock &astBB) {
    Instruction *unreach = BB->getTerminator();
    if (!unreach || !isa<UnreachableInst>(unreach)) return;
    IRBuilder<> B(unreach);
    LC.stateGepCache.clear();
    bool emittedTerminator = false;
    if (astBB.has_terminator()) {
        const lifted_ast::Instruction *termInsn = nullptr;
        for (const auto &ie : astBB.instructions()) {
            if (ie.instruction().id() == astBB.terminator()) {
                termInsn = &ie.instruction(); break;
            }
        }
        if (termInsn) {
            std::string opc = toUpper(termInsn->opcode());
            if (opc == "CALL") {
                if (termInsn->operands_size() > 0) {
                    CallInst* ci = emitFunctionCall(LC, B, termInsn->operands(0), false);
                    if (ci) {
                        Function* callee = ci->getCalledFunction();
                        if ((callee && callee->hasFnAttribute(Attribute::NoReturn)) || astBB.successors_size() == 0) {
                            ci->setDoesNotReturn();
                        }
                    }
                    clobberFlags(LC, B);
                }
                if (astBB.successors_size() == 1 && LC.bbIdToLlvm.count(astBB.successors(0))) {
                    B.CreateBr(LC.bbIdToLlvm[astBB.successors(0)]);
                } else {
                    B.CreateUnreachable();
                }
                emittedTerminator = true;
            }
            else if (opc == "SYSCALL") {
                Function *syscallF = LC.M.getFunction("syscall");
                if (!syscallF) {
                    FunctionType *sft = FunctionType::get(B.getInt64Ty(), {B.getInt64Ty()}, true);
                    syscallF = Function::Create(sft, GlobalValue::ExternalLinkage, "syscall", &LC.M);
                }
                SmallVector<Value*, 7> args;
                args.push_back(loadGpr64(LC, B, gprFieldIndex64("RAX")));
                args.push_back(loadGpr64(LC, B, gprFieldIndex64("RDI")));
                args.push_back(loadGpr64(LC, B, gprFieldIndex64("RSI")));
                args.push_back(loadGpr64(LC, B, gprFieldIndex64("RDX")));
                args.push_back(loadGpr64(LC, B, gprFieldIndex64("R10")));
                args.push_back(loadGpr64(LC, B, gprFieldIndex64("R8")));
                args.push_back(loadGpr64(LC, B, gprFieldIndex64("R9")));
                CallInst *ci = B.CreateCall(syscallF, args);
                storeGpr64(LC, B, gprFieldIndex64("RAX"), ci);
                clobberFlags(LC, B);
                if (astBB.successors_size() == 1 && LC.bbIdToLlvm.count(astBB.successors(0))) {
                    B.CreateBr(LC.bbIdToLlvm[astBB.successors(0)]);
                } else {
                    B.CreateUnreachable();
                }
                emittedTerminator = true;
            }
            else if (opc == "JMP") {
                bool isTailCall = false;
                if (termInsn->operands_size() > 0 && termInsn->operands(0).has_symbol_ref()) {
                    const std::string &sym = termInsn->operands(0).symbol_ref();
                    auto it = LC.P->symbol_table().find(sym);
                    if (it != LC.P->symbol_table().end()) {
                        if (it->second.has_lifted_ref() || it->second.has_wrapper_ref() ||
                            it->second.kind() == "function" || it->second.is_external()) {
                            isTailCall = true;
                        }
                    } else if (LC.M.getFunction(sym)) {
                        isTailCall = true;
                    }
                }
                if (isTailCall) {
                    CallInst* ci = emitFunctionCall(LC, B, termInsn->operands(0), true);
                    bool isNoRet = false;
                    if (ci) {
                        Function* callee = ci->getCalledFunction();
                        if (callee && callee->hasFnAttribute(Attribute::NoReturn)) {
                            ci->setDoesNotReturn();
                            isNoRet = true;
                        }
                    }
                    if (termInsn->operands_size() > 0 && termInsn->operands(0).has_symbol_ref()) {
                        auto it = LC.P->symbol_table().find(termInsn->operands(0).symbol_ref());
                        if (it != LC.P->symbol_table().end() && it->second.has_llvm_type() &&
                            it->second.llvm_type().find("noreturn") != std::string::npos) {
                            isNoRet = true;
                            if (ci) ci->setDoesNotReturn();
                        }
                    }
                    clobberFlags(LC, B);
                    if (isNoRet) {
                        B.CreateUnreachable();
                    } else {
                        Type *rTy = LC.F->getReturnType();
                        if (rTy->isVoidTy()) {
                            B.CreateRetVoid();
                        } else if (rTy->isFloatTy() || rTy->isDoubleTy()) {
                            Value *val = B.CreateExtractElement(loadXmm(LC, B, 0), (uint64_t)0);
                            if (rTy->isFloatTy()) val = B.CreateFPTrunc(val, rTy);
                            B.CreateRet(val);
                        } else {
                            Value *val = loadGpr64(LC, B, gprFieldIndex64("RAX"));
                            if (rTy->isPointerTy()) B.CreateRet(B.CreateIntToPtr(val, rTy));
                            else B.CreateRet(truncOrZext(B, val, rTy));
                        }
                    }
                    emittedTerminator = true;
                }
                else if (termInsn->operands_size() > 0 && (termInsn->operands(0).has_register_() || termInsn->operands(0).has_memory())) {
                    Value *targetPtr = resolveRValue(LC, B, termInsn->operands(0), B.getPtrTy());
                    if (termInsn->indirect_targets_size() > 0) {
                        auto *indbr = B.CreateIndirectBr(targetPtr, termInsn->indirect_targets_size());
                        for (const auto &tgt : termInsn->indirect_targets()) {
                            if (LC.bbIdToLlvm.count(tgt)) indbr->addDestination(LC.bbIdToLlvm[tgt]);
                        }
                    } else {
                        Function *abortF = LC.M.getFunction("abort");
                        if (!abortF) abortF = Function::Create(FunctionType::get(B.getVoidTy(), false), GlobalValue::ExternalLinkage, "abort", &LC.M);
                        B.CreateCall(abortF);
                        B.CreateUnreachable();
                    }
                    emittedTerminator = true;
                } else if (termInsn->target_blocks_size() > 0) {
                    std::string tgt = termInsn->target_blocks(0);
                    if (LC.bbIdToLlvm.count(tgt)) B.CreateBr(LC.bbIdToLlvm[tgt]);
                    else B.CreateUnreachable();
                    emittedTerminator = true;
                } else if (astBB.successors_size() > 0) {
                    std::string tgt = astBB.successors(0);
                    if (LC.bbIdToLlvm.count(tgt)) B.CreateBr(LC.bbIdToLlvm[tgt]);
                    else B.CreateUnreachable();
                    emittedTerminator = true;
                } else {
                    B.CreateUnreachable();
                    emittedTerminator = true;
                }
            }
            else if (opc == "RET" || opc == "IRET" || opc == "SYSRET") {
                if (LC.F->hasFnAttribute(Attribute::NoReturn)) {
                    B.CreateUnreachable();
                } else {
                    Type *rTy = LC.F->getReturnType();
                    if (rTy->isVoidTy()) {
                        B.CreateRetVoid();
                    } else if (rTy->isFloatTy() || rTy->isDoubleTy()) {
                        Value *val = B.CreateExtractElement(loadXmm(LC, B, 0), (uint64_t)0);
                        if (rTy->isFloatTy()) val = B.CreateFPTrunc(val, rTy);
                        B.CreateRet(val);
                    } else {
                        Value *val = loadGpr64(LC, B, gprFieldIndex64("RAX"));
                        if (rTy->isPointerTy()) B.CreateRet(B.CreateIntToPtr(val, rTy));
                        else B.CreateRet(truncOrZext(B, val, rTy));
                    }
                }
                emittedTerminator = true;
            }
            else if (opc.size() >= 2 && opc[0] == 'J') {
                Value *cond = nullptr;
                if (opc == "JE" || opc == "JZ") cond = loadFlag(LC, B, ZF);
                else if (opc == "JNE" || opc == "JNZ") cond = B.CreateNot(loadFlag(LC, B, ZF));
                else if (opc == "JS") cond = loadFlag(LC, B, SF);
                else if (opc == "JNS") cond = B.CreateNot(loadFlag(LC, B, SF));
                else if (opc == "JO") cond = loadFlag(LC, B, OF);
                else if (opc == "JNO") cond = B.CreateNot(loadFlag(LC, B, OF));
                else if (opc == "JP" || opc == "JPE") cond = loadFlag(LC, B, PF);
                else if (opc == "JNP" || opc == "JPO") cond = B.CreateNot(loadFlag(LC, B, PF));
                else if (opc == "JB" || opc == "JC" || opc == "JNAE") cond = loadFlag(LC, B, CF);
                else if (opc == "JAE" || opc == "JNB" || opc == "JNC") cond = B.CreateNot(loadFlag(LC, B, CF));
                else if (opc == "JBE" || opc == "JNA") cond = B.CreateOr(loadFlag(LC, B, CF), loadFlag(LC, B, ZF));
                else if (opc == "JA" || opc == "JNBE") cond = B.CreateAnd(B.CreateNot(loadFlag(LC, B, CF)), B.CreateNot(loadFlag(LC, B, ZF)));
                else if (opc == "JL" || opc == "JNGE") cond = B.CreateICmpNE(loadFlag(LC, B, SF), loadFlag(LC, B, OF));
                else if (opc == "JGE" || opc == "JNL") cond = B.CreateICmpEQ(loadFlag(LC, B, SF), loadFlag(LC, B, OF));
                else if (opc == "JLE" || opc == "JNG") cond = B.CreateOr(loadFlag(LC, B, ZF), B.CreateICmpNE(loadFlag(LC, B, SF), loadFlag(LC, B, OF)));
                else if (opc == "JG" || opc == "JNLE") cond = B.CreateAnd(B.CreateNot(loadFlag(LC, B, ZF)), B.CreateICmpEQ(loadFlag(LC, B, SF), loadFlag(LC, B, OF)));
                else cond = B.getFalse();
                std::string trueBBId = termInsn->target_blocks_size() > 0 ? termInsn->target_blocks(0) : "";
                std::string falseBBId = "";
                for (const auto &succ : astBB.successors()) {
                    if (succ != trueBBId) { falseBBId = succ; break; }
                }
                if (falseBBId.empty() && astBB.successors_size() > 0) {
                    falseBBId = astBB.successors(astBB.successors_size() - 1);
                }
                if (!trueBBId.empty() && !falseBBId.empty() && LC.bbIdToLlvm.count(trueBBId) && LC.bbIdToLlvm.count(falseBBId)) {
                    B.CreateCondBr(cond, LC.bbIdToLlvm[trueBBId], LC.bbIdToLlvm[falseBBId]);
                } else {
                    B.CreateUnreachable();
                }
                emittedTerminator = true;
            }
            else if (opc == "LOOP" || opc == "LOOPE" || opc == "LOOPNE") {
                Value *rcx = loadGpr64(LC, B, gprFieldIndex64("RCX"));
                Value *dec = B.CreateSub(rcx, B.getInt64(1));
                storeGpr64(LC, B, gprFieldIndex64("RCX"), dec);
                Value *cond = B.CreateICmpNE(dec, B.getInt64(0));
                if (opc == "LOOPE") cond = B.CreateAnd(cond, loadFlag(LC, B, ZF));
                if (opc == "LOOPNE") cond = B.CreateAnd(cond, B.CreateNot(loadFlag(LC, B, ZF)));
                std::string trueBBId = termInsn->target_blocks_size() > 0 ? termInsn->target_blocks(0) : "";
                std::string falseBBId = "";
                for (const auto &succ : astBB.successors()) {
                    if (succ != trueBBId) { falseBBId = succ; break; }
                }
                if (falseBBId.empty() && astBB.successors_size() > 0) {
                    falseBBId = astBB.successors(astBB.successors_size() - 1);
                }
                if (!trueBBId.empty() && !falseBBId.empty() && LC.bbIdToLlvm.count(trueBBId) && LC.bbIdToLlvm.count(falseBBId)) {
                    B.CreateCondBr(cond, LC.bbIdToLlvm[trueBBId], LC.bbIdToLlvm[falseBBId]);
                } else {
                    B.CreateUnreachable();
                }
                emittedTerminator = true;
            }
            else if (opc == "HLT") {
                B.CreateUnreachable();
                emittedTerminator = true;
            }
        }
    }
    if (!emittedTerminator) {
        if (astBB.successors_size() == 1) {
            std::string succId = astBB.successors(0);
            if (LC.bbIdToLlvm.count(succId)) {
                B.CreateBr(LC.bbIdToLlvm[succId]);
            } else {
                B.CreateUnreachable();
            }
        } else {
            B.CreateUnreachable();
        }
    }
    unreach->eraseFromParent();
}

static void processFunction(FnLowerCtx &LC) {
    if (!LC.F || LC.F->empty()) return;
    std::map<std::string, const lifted_ast::Instruction*> idToAstInsn;
    for (const auto &bbAst : LC.FnAst->basic_blocks()) {
        std::string bbName = bbAst.has_start_label() && !bbAst.start_label().empty() ? bbAst.start_label() : (bbAst.has_id() && !bbAst.id().empty() ? bbAst.id() : "");
        if (!bbName.empty() && LC.bbIdToLlvm.count(bbName)) {
            LC.bbIdToLlvm[bbAst.id()] = LC.bbIdToLlvm[bbName];
        }
        for (const auto &ie : bbAst.instructions()) {
            if (ie.instruction().has_id()) idToAstInsn[ie.instruction().id()] = &ie.instruction();
        }
    }
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
            if (opc == "CALL") lowerCall(LC, CI, *astIt->second);
            else if (opc == "SYSCALL") lowerSyscall(LC, CI, *astIt->second);
        }
    }
    for (const auto &bbAst : LC.FnAst->basic_blocks()) {
        if (!bbAst.has_id() || !LC.bbIdToLlvm.count(bbAst.id())) continue;
        BasicBlock *BB = LC.bbIdToLlvm[bbAst.id()];
        lowerTerminator(LC, BB, bbAst);
    }
}

// ----------------------------- Main Pipeline Driver -----------------------------
static std::unique_ptr<Module> loadBitcodeModule(const std::string &path, LLVMContext &C) {
    auto bufOrErr = MemoryBuffer::getFile(path);
    if (!bufOrErr) return nullptr;
    auto modOrErr = parseBitcodeFile(bufOrErr->get()->getMemBufferRef(), C);
    if (!modOrErr) return nullptr;
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
    if (args.size() < 4) {
        std::cerr << "Usage: llvmLowerControlFlow <in.bc> <in.pb> <out.bc> <out.pb> [--print-ir out.ll] [--print]\n";
        return 1;
    }
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
                FnLowerCtx LC{C, *M, StateTy, F, nullptr, &P, &fnAst, instrLlvmMapping, nullptr};
                for (auto &BB : *F) LC.bbIdToLlvm[BB.getName().str()] = &BB;
                if (!F->empty()) LC.StateArg = F->getArg(0);
                processFunction(LC);
            }
        }
    }
    for (const auto &kv : instrLlvmMapping) (*P.mutable_instr_llvm_mapping())[kv.first] = kv.second;
    if (verifyModule(*M, &errs())) {
        errs() << "LLVM module verification FAILED after Step 13 lowering.\n";
    }
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
