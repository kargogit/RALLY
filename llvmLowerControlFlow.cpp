/*
llvmLowerControlFlow.cpp

Supported Instructions:

Control Flow
- CALL
- JMP (direct + indirect + tail-call variants)
- RET
- IRET
- SYSRET
- SYSCALL

Conditional Jumps (Jcc family)
- JE/JZ, JNE/JNZ
- JS, JNS
- JO, JNO
- JP/JPE, JNP/JPO
- JB/JC/JNAE, JAE/JNB/JNC
- JBE/JNA, JA/JNBE
- JL/JNGE, JGE/JNL
- JLE/JNG, JG/JNLE

Loop Instructions
- LOOP, LOOPE, LOOPNE
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

// ----------------------------- Utilities -----------------------------

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

// ----------------------------- %State Layout -----------------------------

static constexpr unsigned kGprCount  = 16;
static constexpr unsigned kFlagCount = 9;
static constexpr unsigned kXmmCount  = 16;
static constexpr unsigned kFlagsBase = kGprCount;
static constexpr unsigned kXmmBase   = kGprCount + kFlagCount;

enum FlagIndex : unsigned {
    CF = 0, PF = 1, AF = 2, ZF = 3, SF = 4, OF = 5, DF = 6, IF_FLAG = 7, RF = 8,
};

// ----------------------------- Register & Context -----------------------------

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
        auto nOpt = parseInt64Loose(r.substr(3));
        if (nOpt && *nOpt >= 0 && *nOpt < kXmmCount) {
            ri.isValid = true; ri.isXmm = true; ri.xmmIndex = (unsigned)*nOpt;
        }
        return ri;
    }

    auto setGpr = [&](const std::string &base64, unsigned w, unsigned off) {
        int idx = gprFieldIndex64(base64);
        if (idx >= 0) { ri.isValid = true; ri.gprIndex = idx; ri.bitWidth = w; ri.bitOffset = off; }
    };

    if (r == "RAX") { setGpr("RAX",64,0); } else if (r == "EAX") { setGpr("RAX",32,0); } else if (r == "AX") { setGpr("RAX",16,0); } else if (r == "AL") { setGpr("RAX",8,0); } else if (r == "AH") { setGpr("RAX",8,8); }
    else if (r == "RBX") { setGpr("RBX",64,0); } else if (r == "EBX") { setGpr("RBX",32,0); } else if (r == "BX") { setGpr("RBX",16,0); } else if (r == "BL") { setGpr("RBX",8,0); } else if (r == "BH") { setGpr("RBX",8,8); }
    else if (r == "RCX") { setGpr("RCX",64,0); } else if (r == "ECX") { setGpr("RCX",32,0); } else if (r == "CX") { setGpr("RCX",16,0); } else if (r == "CL") { setGpr("RCX",8,0); } else if (r == "CH") { setGpr("RCX",8,8); }
    else if (r == "RDX") { setGpr("RDX",64,0); } else if (r == "EDX") { setGpr("RDX",32,0); } else if (r == "DX") { setGpr("RDX",16,0); } else if (r == "DL") { setGpr("RDX",8,0); } else if (r == "DH") { setGpr("RDX",8,8); }
    else if (r == "RSI") { setGpr("RSI",64,0); } else if (r == "ESI") { setGpr("RSI",32,0); } else if (r == "SI") { setGpr("RSI",16,0); } else if (r == "SIL") { setGpr("RSI",8,0); }
    else if (r == "RDI") { setGpr("RDI",64,0); } else if (r == "EDI") { setGpr("RDI",32,0); } else if (r == "DI") { setGpr("RDI",16,0); } else if (r == "DIL") { setGpr("RDI",8,0); }
    else if (r == "RBP") { setGpr("RBP",64,0); } else if (r == "EBP") { setGpr("RBP",32,0); } else if (r == "BP") { setGpr("RBP",16,0); } else if (r == "BPL") { setGpr("RBP",8,0); }
    else if (r == "RSP") { setGpr("RSP",64,0); } else if (r == "ESP") { setGpr("RSP",32,0); } else if (r == "SP") { setGpr("RSP",16,0); } else if (r == "SPL") { setGpr("RSP",8,0); }
    else if (r.size() >= 2 && r[0] == 'R' && std::isdigit(r[1])) {
        std::string base = "R";
        size_t i = 1; while (i < r.size() && std::isdigit(r[i])) { base.push_back(r[i]); ++i; }
        std::string suff = r.substr(i);
        if (suff.empty()) setGpr(base,64,0); else if (suff == "D") setGpr(base,32,0); else if (suff == "W") setGpr(base,16,0); else if (suff == "B") setGpr(base,8,0);
    }
    return ri;
}

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
        : C(c), M(m), StateTy(stateTy), F(f), StateArg(stateArg), P(p), FnAst(fnAst), instrLlvmMapping(instrMap) {}
};

static llvm::Value* getStateFieldPtr(FnLowerCtx &LC, IRBuilder<> &B, unsigned fieldIdx) {
    auto it = LC.stateGepCache.find(fieldIdx);
    if (it != LC.stateGepCache.end()) return it->second;
    llvm::Value *p = B.CreateStructGEP(LC.StateTy, LC.StateArg, fieldIdx, "state.gep");
    LC.stateGepCache[fieldIdx] = p;
    return p;
}

static llvm::Value* loadGpr64(FnLowerCtx &LC, IRBuilder<> &B, unsigned gprIdx) {
    return B.CreateLoad(B.getInt64Ty(), getStateFieldPtr(LC, B, gprIdx), "gpr64");
}

static void storeGpr64(FnLowerCtx &LC, IRBuilder<> &B, unsigned gprIdx, llvm::Value *vI64) {
    B.CreateStore(vI64, getStateFieldPtr(LC, B, gprIdx));
}

static llvm::Value* loadFlag(FnLowerCtx &LC, IRBuilder<> &B, FlagIndex f) {
    return B.CreateLoad(B.getInt1Ty(), getStateFieldPtr(LC, B, kFlagsBase + (unsigned)f), "flag");
}

static void storeFlag(FnLowerCtx &LC, IRBuilder<> &B, FlagIndex f, llvm::Value *vI1) {
    B.CreateStore(vI1, getStateFieldPtr(LC, B, kFlagsBase + (unsigned)f));
}

static llvm::Value* loadXmm(FnLowerCtx &LC, IRBuilder<> &B, unsigned xmmIdx) {
    return B.CreateLoad(llvm::FixedVectorType::get(B.getDoubleTy(), 2), getStateFieldPtr(LC, B, kXmmBase + xmmIdx), "xmm.ld");
}

static void storeXmm(FnLowerCtx &LC, IRBuilder<> &B, unsigned xmmIdx, llvm::Value *vec) {
    B.CreateStore(vec, getStateFieldPtr(LC, B, kXmmBase + xmmIdx));
}

static llvm::Type* intTy(llvm::LLVMContext &C, unsigned bits) {
    return llvm::IntegerType::get(C, bits);
}

static llvm::Value* truncOrZext(IRBuilder<> &B, llvm::Value *v, llvm::Type *dstTy) {
    llvm::Type *srcTy = v->getType();
    if (srcTy == dstTy) return v;
    if (srcTy->isIntegerTy() && dstTy->isIntegerTy()) {
        unsigned sb = srcTy->getIntegerBitWidth(), db = dstTy->getIntegerBitWidth();
        return (sb > db) ? B.CreateTrunc(v, dstTy) : B.CreateZExt(v, dstTy);
    }
    return B.CreateBitCast(v, dstTy);
}

static llvm::Value* readGprSubreg(FnLowerCtx &LC, IRBuilder<> &B, const RegInfo &ri) {
    llvm::Value *full = loadGpr64(LC, B, (unsigned)ri.gprIndex);
    if (ri.bitWidth == 64 && ri.bitOffset == 0) return full;
    llvm::Value *shifted = ri.bitOffset != 0 ? B.CreateLShr(full, B.getInt64(ri.bitOffset)) : full;
    return B.CreateTrunc(shifted, intTy(LC.C, ri.bitWidth));
}

static void clobberFlags(FnLowerCtx &LC, IRBuilder<> &B) {
    llvm::Value *undefFlag = llvm::UndefValue::get(B.getInt1Ty());
    for(unsigned i = 0; i < kFlagCount; ++i) {
        storeFlag(LC, B, static_cast<FlagIndex>(i), undefFlag);
    }
}

// ----------------------------- Memory / Address Evaluators -----------------------------

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

static llvm::Value* resolveMemAddress(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Operand &op) {
    if (op.has_symbol_ref()) {
        llvm::Value *basePtr = symbolAddressAsPtr(LC, B, op.symbol_ref());
        if (op.has_memory() && op.memory().has_displacement()) {
            if (auto disp = valueAsInt64(op.memory().displacement()); disp && *disp != 0) {
                basePtr = B.CreateGEP(B.getInt8Ty(), basePtr, B.getInt64(*disp), "sym.disp");
            }
        }
        return basePtr;
    }
    llvm::Value *addrI64 = B.getInt64(0);
    if (op.has_memory()) {
        const auto &m = op.memory();
        if (m.has_base() && toUpper(m.base()) != "RIP") {
            RegInfo bri = decodeReg(m.base());
            if (bri.isValid && !bri.isXmm) addrI64 = B.CreateAdd(addrI64, truncOrZext(B, readGprSubreg(LC, B, bri), B.getInt64Ty()));
        }
        if (m.has_index()) {
            RegInfo iri = decodeReg(m.index());
            if (iri.isValid && !iri.isXmm) {
                llvm::Value *idx = truncOrZext(B, readGprSubreg(LC, B, iri), B.getInt64Ty());
                if (m.has_scale() && m.scale() > 1) idx = B.CreateMul(idx, B.getInt64(m.scale()));
                addrI64 = B.CreateAdd(addrI64, idx);
            }
        }
        if (m.has_displacement()) {
            if (auto d = valueAsInt64(m.displacement()); d && *d != 0) addrI64 = B.CreateAdd(addrI64, B.getInt64(*d));
        }
    }
    return B.CreateIntToPtr(addrI64, B.getPtrTy());
}

static llvm::Value* resolveRValue(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Operand &op, llvm::Type *desiredTy) {
    if (op.has_register_()) {
        RegInfo ri = decodeReg(op.register_());
        if (ri.isValid && !ri.isXmm) {
            llvm::Value *v = readGprSubreg(LC, B, ri);
            if (desiredTy->isPointerTy()) return B.CreateIntToPtr(truncOrZext(B, v, B.getInt64Ty()), desiredTy);
            return truncOrZext(B, v, desiredTy);
        }
    }
    if (op.has_memory() || op.has_symbol_ref()) {
        return resolveMemAddress(LC, B, op);
    }
    return llvm::UndefValue::get(desiredTy);
}

// ----------------------------- Erasure Utils -----------------------------

static std::optional<std::string> getPlaceholderOpcode(llvm::CallInst *CI) {
    if (!CI) return std::nullopt;
    if (auto *IA = llvm::dyn_cast<llvm::InlineAsm>(CI->getCalledOperand())) {
        std::string asmStr = IA->getAsmString();
        const char *prefix = "; placeholder for ";
        if (size_t pos = asmStr.find(prefix); pos != std::string::npos) {
            std::string opc = asmStr.substr(pos + strlen(prefix));
            while (!opc.empty() && std::isspace((unsigned char)opc.back())) opc.pop_back();
            return opc;
        }
    }
    return std::nullopt;
}

static std::string getAstInstrId(llvm::Instruction *I) {
    if (I) {
        if (llvm::MDNode *md = I->getMetadata("ast_instr_id")) {
            if (md->getNumOperands() > 0) {
                if (auto *str = llvm::dyn_cast<llvm::MDString>(md->getOperand(0))) return str->getString().str();
            }
        }
    }
    return "";
}

static void eraseDummyStoresAfter(llvm::Instruction *start, const std::string &placeholderInstrId) {
    llvm::SmallVector<llvm::Instruction*, 8> toErase;
    llvm::Instruction *cur = start;
    while (cur && !cur->isTerminator()) {
        if (auto *SI = llvm::dyn_cast<llvm::StoreInst>(cur)) {
            std::string sid = getAstInstrId(SI);
            if (!sid.empty() && sid != placeholderInstrId) break;
            llvm::Value *val = SI->getValueOperand();
            bool isDummy = llvm::isa<llvm::UndefValue>(val) || llvm::isa<llvm::ConstantAggregateZero>(val) || (llvm::isa<llvm::ConstantInt>(val) && llvm::cast<llvm::ConstantInt>(val)->isZero());
            if (isDummy) {
                toErase.push_back(cur);
                cur = cur->getNextNode();
                if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(SI->getPointerOperand())) {
                    if (GEP->hasOneUse()) toErase.push_back(GEP);
                }
                continue;
            }
        }
        if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(cur)) {
            std::string gid = getAstInstrId(GEP);
            if (gid.empty() || gid == placeholderInstrId) {
                if (llvm::Instruction *next = cur->getNextNode()) {
                    if (auto *SI = llvm::dyn_cast<llvm::StoreInst>(next)) {
                        if (SI->getPointerOperand() == GEP) {
                            llvm::Value *val = SI->getValueOperand();
                            bool isDummy = llvm::isa<llvm::UndefValue>(val) || llvm::isa<llvm::ConstantAggregateZero>(val) || (llvm::isa<llvm::ConstantInt>(val) && llvm::cast<llvm::ConstantInt>(val)->isZero());
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

    llvm::SmallVector<llvm::Instruction*, 4> stores, geps;
    for (auto *I : toErase) { if (llvm::isa<llvm::StoreInst>(I)) stores.push_back(I); else geps.push_back(I); }
    for (auto *I : stores) I->eraseFromParent();
    for (auto *I : geps) { if (I->use_empty()) I->eraseFromParent(); }
}

// ----------------------------- Step 13 Specific Lowering -----------------------------

// Helper to consistently construct the LLVM IR representation for a function call target
static llvm::CallInst* emitFunctionCall(FnLowerCtx &LC, IRBuilder<> &B, const lifted_ast::Operand &op, bool isTailCall) {
    bool isInternal = false;
    llvm::Function *targetF = nullptr;

    if (op.has_symbol_ref()) {
        auto it = LC.P->symbol_table().find(op.symbol_ref());
        if (it != LC.P->symbol_table().end() && it->second.has_lifted_ref() && !it->second.lifted_ref().empty()) {
            isInternal = true;
            targetF = LC.M.getFunction(it->second.lifted_ref());
        } else {
            targetF = LC.M.getFunction(op.symbol_ref());
        }
    }

    llvm::CallInst* ci = nullptr;

    if (isInternal && targetF) {
        ci = B.CreateCall(targetF, {LC.StateArg});
    }
    else if (!isInternal && targetF) {
        llvm::FunctionType *ft = targetF->getFunctionType();
        llvm::SmallVector<llvm::Value*, 8> args;

        const int intRegs[] = { gprFieldIndex64("RDI"), gprFieldIndex64("RSI"), gprFieldIndex64("RDX"),
                                gprFieldIndex64("RCX"), gprFieldIndex64("R8"), gprFieldIndex64("R9") };
        int intIdx = 0, fpIdx = 0;

        for (llvm::Type *paramTy : ft->params()) {
            if (paramTy->isFloatTy() || paramTy->isDoubleTy()) {
                llvm::Value *vec = loadXmm(LC, B, fpIdx < 8 ? fpIdx : 7);
                llvm::Value *val = B.CreateExtractElement(vec, (uint64_t)0);
                if (paramTy->isFloatTy()) val = B.CreateFPTrunc(val, paramTy);
                args.push_back(val);
                fpIdx++;
            } else {
                llvm::Value *val = loadGpr64(LC, B, intIdx < 6 ? intRegs[intIdx] : 0);
                if (paramTy->isPointerTy()) val = B.CreateIntToPtr(val, paramTy);
                else val = truncOrZext(B, val, paramTy);
                args.push_back(val);
                intIdx++;
            }
        }

        if (ft->isVarArg()) {
            while(intIdx < 6) { args.push_back(loadGpr64(LC, B, intRegs[intIdx++])); }
            while(fpIdx < 8) { args.push_back(B.CreateExtractElement(loadXmm(LC, B, fpIdx++), (uint64_t)0)); }
        }

        ci = B.CreateCall(targetF, args);

        if (!ft->getReturnType()->isVoidTy()) {
            llvm::Type *rTy = ft->getReturnType();
            if (rTy->isFloatTy() || rTy->isDoubleTy()) {
                llvm::Value *val = ci;
                if (rTy->isFloatTy()) val = B.CreateFPExt(val, B.getDoubleTy());
                llvm::Value *vec = llvm::UndefValue::get(llvm::FixedVectorType::get(B.getDoubleTy(), 2));
                vec = B.CreateInsertElement(vec, val, (uint64_t)0);
                vec = B.CreateInsertElement(vec, llvm::ConstantFP::get(B.getDoubleTy(), 0.0), (uint64_t)1);
                storeXmm(LC, B, 0, vec);
            } else {
                llvm::Value *val = ci;
                if (rTy->isPointerTy()) val = B.CreatePtrToInt(val, B.getInt64Ty());
                else val = truncOrZext(B, val, B.getInt64Ty());
                storeGpr64(LC, B, 0, val);
            }
        }
    }
    else {
        llvm::Value *targetPtr = resolveRValue(LC, B, op, B.getPtrTy());
        llvm::FunctionType *ft = llvm::FunctionType::get(B.getVoidTy(), {}, false);
        ci = B.CreateCall(ft, targetPtr, {});
    }

    if (ci && isTailCall) {
        ci->setTailCall(true);
    }
    return ci;
}

static void lowerCall(FnLowerCtx &LC, llvm::CallInst *placeholder, const lifted_ast::Instruction &astInsn) {
    IRBuilder<> B(placeholder);
    LC.stateGepCache.clear();

    llvm::Instruction* nextInstr = placeholder->getNextNode();
    std::string instrId = getAstInstrId(placeholder);

    if (astInsn.operands_size() > 0) {
        emitFunctionCall(LC, B, astInsn.operands(0), false);
    }

    clobberFlags(LC, B);
    placeholder->eraseFromParent();
    eraseDummyStoresAfter(nextInstr, instrId);
}

static void lowerSyscall(FnLowerCtx &LC, llvm::CallInst *placeholder, const lifted_ast::Instruction &astInsn) {
    IRBuilder<> B(placeholder);
    LC.stateGepCache.clear();

    llvm::Instruction* nextInstr = placeholder->getNextNode();
    std::string instrId = getAstInstrId(placeholder);

    llvm::Function *syscallF = LC.M.getFunction("syscall");
    if (!syscallF) {
        llvm::FunctionType *sft = llvm::FunctionType::get(B.getInt64Ty(), {B.getInt64Ty()}, true);
        syscallF = llvm::Function::Create(sft, llvm::GlobalValue::ExternalLinkage, "syscall", &LC.M);
    }

    llvm::SmallVector<llvm::Value*, 7> args;
    args.push_back(loadGpr64(LC, B, gprFieldIndex64("RAX"))); // sysno
    args.push_back(loadGpr64(LC, B, gprFieldIndex64("RDI")));
    args.push_back(loadGpr64(LC, B, gprFieldIndex64("RSI")));
    args.push_back(loadGpr64(LC, B, gprFieldIndex64("RDX")));
    args.push_back(loadGpr64(LC, B, gprFieldIndex64("R10")));
    args.push_back(loadGpr64(LC, B, gprFieldIndex64("R8")));
    args.push_back(loadGpr64(LC, B, gprFieldIndex64("R9")));

    llvm::CallInst *ci = B.CreateCall(syscallF, args);
    storeGpr64(LC, B, gprFieldIndex64("RAX"), ci);

    clobberFlags(LC, B);
    placeholder->eraseFromParent();
    eraseDummyStoresAfter(nextInstr, instrId);
}

static void lowerTerminator(FnLowerCtx &LC, llvm::BasicBlock *BB, const lifted_ast::BasicBlock &astBB) {
    llvm::Instruction *unreach = BB->getTerminator();
    if (!unreach || !llvm::isa<llvm::UnreachableInst>(unreach)) return;

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
                    llvm::CallInst* ci = emitFunctionCall(LC, B, termInsn->operands(0), false);
                    if (ci) {
                        llvm::Function* callee = ci->getCalledFunction();
                        // Either identified directly as noreturn via Attributes, or dynamically via CFG zero-successors layout
                        if ((callee && callee->hasFnAttribute(llvm::Attribute::NoReturn)) || astBB.successors_size() == 0) {
                            ci->setDoesNotReturn();
                        }
                    }
                    clobberFlags(LC, B);
                }

                // Strictly conform to successors mapping. If it genuinely doesn't return, successors_size == 0
                if (astBB.successors_size() == 1 && LC.bbIdToLlvm.count(astBB.successors(0))) {
                    B.CreateBr(LC.bbIdToLlvm[astBB.successors(0)]);
                } else {
                    B.CreateUnreachable();
                }
                emittedTerminator = true;
            }
            else if (opc == "SYSCALL") {
                llvm::Function *syscallF = LC.M.getFunction("syscall");
                if (!syscallF) {
                    llvm::FunctionType *sft = llvm::FunctionType::get(B.getInt64Ty(), {B.getInt64Ty()}, true);
                    syscallF = llvm::Function::Create(sft, llvm::GlobalValue::ExternalLinkage, "syscall", &LC.M);
                }

                llvm::SmallVector<llvm::Value*, 7> args;
                args.push_back(loadGpr64(LC, B, gprFieldIndex64("RAX")));
                args.push_back(loadGpr64(LC, B, gprFieldIndex64("RDI")));
                args.push_back(loadGpr64(LC, B, gprFieldIndex64("RSI")));
                args.push_back(loadGpr64(LC, B, gprFieldIndex64("RDX")));
                args.push_back(loadGpr64(LC, B, gprFieldIndex64("R10")));
                args.push_back(loadGpr64(LC, B, gprFieldIndex64("R8")));
                args.push_back(loadGpr64(LC, B, gprFieldIndex64("R9")));

                llvm::CallInst *ci = B.CreateCall(syscallF, args);
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

                // Identify if this is an interprocedural jump directly mapping to a known external/lifted function
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
                    llvm::CallInst* ci = emitFunctionCall(LC, B, termInsn->operands(0), true);
                    bool isNoRet = false;

                    if (ci) {
                        llvm::Function* callee = ci->getCalledFunction();
                        if (callee && callee->hasFnAttribute(llvm::Attribute::NoReturn)) {
                            ci->setDoesNotReturn();
                            isNoRet = true;
                        }
                    }

                    // Secondary safety check for non-returning nature by querying internal AST's structural metadata
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
                        // For valid returning tail calls, synthesize a block exit using standard return marshaling
                        llvm::Type *rTy = LC.F->getReturnType();
                        if (rTy->isVoidTy()) {
                            B.CreateRetVoid();
                        } else if (rTy->isFloatTy() || rTy->isDoubleTy()) {
                            llvm::Value *val = B.CreateExtractElement(loadXmm(LC, B, 0), (uint64_t)0);
                            if (rTy->isFloatTy()) val = B.CreateFPTrunc(val, rTy);
                            B.CreateRet(val);
                        } else {
                            llvm::Value *val = loadGpr64(LC, B, gprFieldIndex64("RAX"));
                            if (rTy->isPointerTy()) B.CreateRet(B.CreateIntToPtr(val, rTy));
                            else B.CreateRet(truncOrZext(B, val, rTy));
                        }
                    }
                    emittedTerminator = true;
                }
                else if (termInsn->operands_size() > 0 && (termInsn->operands(0).has_register_() || termInsn->operands(0).has_memory())) {
                    llvm::Value *targetPtr = resolveRValue(LC, B, termInsn->operands(0), B.getPtrTy());
                    if (termInsn->indirect_targets_size() > 0) {
                        auto *indbr = B.CreateIndirectBr(targetPtr, termInsn->indirect_targets_size());
                        for (const auto &tgt : termInsn->indirect_targets()) {
                            if (LC.bbIdToLlvm.count(tgt)) indbr->addDestination(LC.bbIdToLlvm[tgt]);
                        }
                    } else {
                        llvm::Function *abortF = LC.M.getFunction("abort");
                        if (!abortF) abortF = llvm::Function::Create(llvm::FunctionType::get(B.getVoidTy(), false), llvm::GlobalValue::ExternalLinkage, "abort", &LC.M);
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
                if (LC.F->hasFnAttribute(llvm::Attribute::NoReturn)) {
                    B.CreateUnreachable();
                } else {
                    llvm::Type *rTy = LC.F->getReturnType();
                    if (rTy->isVoidTy()) {
                        B.CreateRetVoid();
                    } else if (rTy->isFloatTy() || rTy->isDoubleTy()) {
                        llvm::Value *val = B.CreateExtractElement(loadXmm(LC, B, 0), (uint64_t)0);
                        if (rTy->isFloatTy()) val = B.CreateFPTrunc(val, rTy);
                        B.CreateRet(val);
                    } else {
                        llvm::Value *val = loadGpr64(LC, B, gprFieldIndex64("RAX"));
                        if (rTy->isPointerTy()) B.CreateRet(B.CreateIntToPtr(val, rTy));
                        else B.CreateRet(truncOrZext(B, val, rTy));
                    }
                }
                emittedTerminator = true;
            }
            else if (opc.size() >= 2 && opc[0] == 'J') {
                llvm::Value *cond = nullptr;
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
                llvm::Value *rcx = loadGpr64(LC, B, gprFieldIndex64("RCX"));
                llvm::Value *dec = B.CreateSub(rcx, B.getInt64(1));
                storeGpr64(LC, B, gprFieldIndex64("RCX"), dec);

                llvm::Value *cond = B.CreateICmpNE(dec, B.getInt64(0));
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
    LC.StateArg = LC.F->getArg(0);

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

            if (opc == "CALL") lowerCall(LC, CI, *astIt->second);
            else if (opc == "SYSCALL") lowerSyscall(LC, CI, *astIt->second);
        }
    }

    for (const auto &bbAst : LC.FnAst->basic_blocks()) {
        if (!bbAst.has_id() || !LC.bbIdToLlvm.count(bbAst.id())) continue;
        llvm::BasicBlock *BB = LC.bbIdToLlvm[bbAst.id()];
        lowerTerminator(LC, BB, bbAst);
    }
}

// ----------------------------- Main Pipeline Driver -----------------------------

static std::unique_ptr<llvm::Module> loadBitcodeModule(const std::string &path, llvm::LLVMContext &C) {
    auto bufOrErr = llvm::MemoryBuffer::getFile(path);
    if (!bufOrErr) return nullptr;
    auto modOrErr = llvm::parseBitcodeFile(bufOrErr->get()->getMemBufferRef(), C);
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
                for (auto &BB : *F) LC.bbIdToLlvm[BB.getName().str()] = &BB;
                processFunction(LC);
            }
        }
    }

    for (const auto &kv : instrLlvmMapping) (*P.mutable_instr_llvm_mapping())[kv.first] = kv.second;

    if (llvm::verifyModule(*M, &llvm::errs())) {
        llvm::errs() << "LLVM module verification FAILED after Step 13 lowering.\n";
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
