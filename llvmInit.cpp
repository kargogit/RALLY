// llvmInit.cpp (Implementation of Step 10)
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cstdint>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

#include <google/protobuf/util/json_util.h>

#include "ast.pb.h"

using namespace llvm;

static std::string trim(std::string s) {
  auto notSpace = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), notSpace));
  s.erase(std::find_if(s.rbegin(), s.rend(), notSpace).base(), s.end());
  return s;
}

static bool startsWith(const std::string& s, const char* pref) {
  return s.rfind(pref, 0) == 0;
}

static bool endsWith(const std::string& s, const char* suff) {
  size_t n = std::strlen(suff);
  return s.size() >= n && s.compare(s.size() - n, n, suff) == 0;
}

static bool isHex(unsigned char c) {
  return std::isxdigit(c) != 0;
}

static uint8_t hexVal(unsigned char c) {
  if (c >= '0' && c <= '9') return uint8_t(c - '0');
  c = unsigned(std::toupper(c));
  return uint8_t(10 + (c - 'A'));
}

// Decode LLVM IR c"..." content where bytes are encoded as \XX hex.
static std::vector<uint8_t> decodeLlvmCStringContent(const std::string& content) {
  std::vector<uint8_t> out;
  out.reserve(content.size());
  for (size_t i = 0; i < content.size(); ) {
    unsigned char c = (unsigned char)content[i];
    if (c == '\\' && i + 2 < content.size() &&
        isHex((unsigned char)content[i + 1]) && isHex((unsigned char)content[i + 2])) {
      uint8_t b = uint8_t((hexVal((unsigned char)content[i + 1]) << 4) |
                          (hexVal((unsigned char)content[i + 2])));
      out.push_back(b);
      i += 3;
      continue;
    }
    out.push_back(uint8_t(content[i]));
    ++i;
  }
  return out;
}

// Parses simple non-function LLVM types used in your SymbolTable.
static Type* parseLLVMType(LLVMContext& ctx, const std::string& s0) {
  std::string s = trim(s0);

  if (s == "void") return Type::getVoidTy(ctx);
  if (s == "i1") return Type::getInt1Ty(ctx);
  if (s == "i8") return Type::getInt8Ty(ctx);
  if (s == "i32") return Type::getInt32Ty(ctx);
  if (s == "i64") return Type::getInt64Ty(ctx);
  if (s == "float" || s == "f32") return Type::getFloatTy(ctx);
  if (s == "double" || s == "f64") return Type::getDoubleTy(ctx);
  if (s == "ptr") return PointerType::getUnqual(ctx);

  // Treat any T* as opaque ptr.
  if (!s.empty() && s.back() == '*') return PointerType::getUnqual(ctx);
  if (s.find("ptr") != std::string::npos) return PointerType::getUnqual(ctx);

  // Arrays like [15 x i8]
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

  // Fallback
  return PointerType::getUnqual(ctx);
}

// Parse function types.
static FunctionType* parseFunctionType(LLVMContext& ctx, const std::string& s0) {
  std::string s = trim(s0);
  auto lp = s.find('(');
  auto rp = s.rfind(')');
  if (lp == std::string::npos || rp == std::string::npos || rp < lp) {
    return FunctionType::get(PointerType::getUnqual(ctx), {}, true);
  }

  std::string retStr = trim(s.substr(0, lp));
  std::string argsStr = trim(s.substr(lp + 1, rp - (lp + 1)));

  Type* retTy = parseLLVMType(ctx, retStr);

  SmallVector<Type*, 8> args;
  bool vararg = false;

  if (!argsStr.empty()) {
    size_t start = 0;
    while (start < argsStr.size()) {
      size_t comma = argsStr.find(',', start);
      std::string tok = trim(argsStr.substr(start, comma == std::string::npos ? std::string::npos : (comma - start)));
      if (tok == "...") {
        vararg = true;
      } else if (!tok.empty()) {
        args.push_back(parseLLVMType(ctx, tok));
      }

      if (comma == std::string::npos) break;
      start = comma + 1;
    }
  }

  return FunctionType::get(retTy, args, vararg);
}

// Create a minimal stub body for lifted functions to satisfy verifyModule for InternalLinkage.
static void createStubBody(Function* F) {
  LLVMContext& ctx = F->getContext();
  if (!F->empty()) return;

  BasicBlock* bb = BasicBlock::Create(ctx, "entry", F);
  IRBuilder<> B(bb);

  if (F->hasFnAttribute(Attribute::NoReturn)) {
    B.CreateUnreachable();
    return;
  }

  Type* retTy = F->getReturnType();
  if (retTy->isVoidTy()) {
    B.CreateRetVoid();
  } else {
    B.CreateRet(UndefValue::get(retTy));
  }
}

static int gprIndex(const std::string& r) {
  std::string upper = r;
  std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
  static const std::unordered_map<std::string, int> map = {
    {"RAX",0},{"RBX",1},{"RCX",2},{"RDX",3},{"RSI",4},{"RDI",5},
    {"R8",6},{"R9",7},{"RSP",14},{"RBP",15}
  };
  auto it = map.find(upper);
  return (it != map.end()) ? it->second : -1;
}

// Define the named %State struct exactly as per spec.
static StructType* createStateType(LLVMContext& ctx) {
  SmallVector<Type*, 64> fields;
  // 16 x i64 (GPRs)
  for (int i = 0; i < 16; ++i) fields.push_back(Type::getInt64Ty(ctx));
  // 9 x i1 (Flags)
  for (int i = 0; i < 9; ++i) fields.push_back(Type::getInt1Ty(ctx));
  // 16 x <2 x double> (XMM)
  auto xmmTy = FixedVectorType::get(Type::getDoubleTy(ctx), 2);
  for (int i = 0; i < 16; ++i) fields.push_back(xmmTy);

  return StructType::create(ctx, fields, "State");
}

static void applyLiftedAttributes(Function* F, const lifted_ast::LiftedFunctionSignature& sig) {
  for (const auto& a : sig.attributes()) {
    if (a == "noreturn") F->addFnAttr(Attribute::NoReturn);
  }
}

// Determine the appropriate LLVM linkage for a symbol table entry.
static GlobalValue::LinkageTypes resolveLinkage(const lifted_ast::SymbolEntry& entry) {
  if (entry.has_linkage()) {
    if (entry.linkage() == "internal") return GlobalValue::InternalLinkage;
    if (entry.linkage() == "appending") return GlobalValue::AppendingLinkage;
    return GlobalValue::ExternalLinkage;
  }
  if (entry.has_visibility() && entry.visibility() == "local") {
    return GlobalValue::InternalLinkage;
  }
  return GlobalValue::ExternalLinkage;
}

// Build initializer from SymbolEntry.value.
static Constant* buildInitializer(LLVMContext& ctx, Type* declaredTy, const lifted_ast::SymbolEntry& entry) {
  Type* ty = declaredTy;

  if (!entry.has_value()) {
    return Constant::getNullValue(ty);
  }

  const auto& v = entry.value();

  if (v.has_string_value()) {
    std::string sv = trim(v.string_value());

    if (sv == "zeroinitializer") {
      if (ty->isAggregateType()) return ConstantAggregateZero::get(ty);
      return Constant::getNullValue(ty);
    }

    if (startsWith(sv, "c\"") && sv.size() >= 3 && sv.back() == '"') {
      std::string content = sv.substr(2, sv.size() - 3);
      std::vector<uint8_t> bytes = decodeLlvmCStringContent(content);

      if (auto* arrTy = dyn_cast<ArrayType>(ty)) {
        uint64_t n = arrTy->getNumElements();
        bytes.resize((size_t)n, 0);
        return ConstantDataArray::get(ctx, bytes);
      }

      auto* arrTy = ArrayType::get(Type::getInt8Ty(ctx), bytes.size());
      (void)arrTy;
      return ConstantDataArray::get(ctx, bytes);
    }

    if (!sv.empty() && sv.front() == '[' && sv.back() == ']') {
      if (auto* arrTy = dyn_cast<ArrayType>(ty)) {
        auto* elemTy = arrTy->getElementType();
        uint64_t n = arrTy->getNumElements();

        std::string inside = trim(sv.substr(1, sv.size() - 2));
        std::vector<Constant*> elts;
        elts.reserve((size_t)n);

        size_t start = 0;
        while (start < inside.size() && elts.size() < (size_t)n) {
          size_t comma = inside.find(',', start);
          std::string tok = trim(inside.substr(start, comma == std::string::npos ? std::string::npos : (comma - start)));
          if (!tok.empty()) {
            long long val = std::stoll(tok);
            elts.push_back(ConstantInt::get(elemTy, (uint64_t)val, /*isSigned=*/true));
          }
          if (comma == std::string::npos) break;
          start = comma + 1;
        }

        while (elts.size() < (size_t)n) {
          elts.push_back(Constant::getNullValue(elemTy));
        }
        return ConstantArray::get(arrTy, elts);
      }
    }

    if (ty->isIntegerTy()) {
      long long iv = std::stoll(sv);
      return ConstantInt::get(ty, (uint64_t)iv, /*isSigned=*/true);
    }
    if (ty->isFloatingPointTy()) {
      double dv = std::stod(sv);
      return ConstantFP::get(ty, dv);
    }

    return Constant::getNullValue(ty);
  }

  if (v.has_number_value()) {
    double num = v.number_value();

    if (ty->isIntegerTy()) {
      long long iv = (long long)num;
      return ConstantInt::get(ty, (uint64_t)iv, /*isSigned=*/true);
    }
    if (ty->isFloatingPointTy()) {
      return ConstantFP::get(ty, num);
    }
  }

  return Constant::getNullValue(ty);
}

// Synthesize Boundary Wrapper Body exactly as per Step 10 Point 7.
static void synthesizeBoundaryWrapperBody(LLVMContext& ctx,
                                         StructType* stateTy,
                                         Function* wrapper,
                                         Function* liftedF,
                                         bool wrapperNoReturn) {
  BasicBlock* entryBB = BasicBlock::Create(ctx, "entry", wrapper);
  IRBuilder<> B(entryBB);

  // 1. Allocate State with EXACT 64-byte alignment as per spec.
  AllocaInst* state = B.CreateAlloca(stateTy, nullptr, "state");
  state->setAlignment(Align(64)); // FIXED: Was Align(16)

  // 2. Provide a real host stack for the lifted code (required by spec).
  ArrayType* stackArrayTy = ArrayType::get(Type::getInt8Ty(ctx), 4096);
  AllocaInst* hostStack = B.CreateAlloca(stackArrayTy, nullptr, "host_stack");
  hostStack->setAlignment(Align(16));

  // 3. Compute top address with CreateGEP (stack grows downward).
  Value* stackTop = B.CreateGEP(
      stackArrayTy, hostStack,
      {B.getInt64(0), B.getInt64(4096)},
      "stack_top",
      /*inBounds=*/true
  );
  Value* stackTopI64 = B.CreatePtrToInt(stackTop, Type::getInt64Ty(ctx));

  // 4. Store into %State fields 14 (RSP) and 15 (RBP).
  Value* rspPtr = B.CreateStructGEP(stateTy, state, 14);
  B.CreateStore(stackTopI64, rspPtr);
  Value* rbpPtr = B.CreateStructGEP(stateTy, state, 15);
  B.CreateStore(stackTopI64, rbpPtr);

  // 5. Marshal ABI args (SysV AMD64).
  constexpr unsigned kGprCount = 16;
  constexpr unsigned kFlagCount = 9;
  constexpr unsigned kXmmBase = kGprCount + kFlagCount;
  const int gprOrder[6] = {
    gprIndex("RDI"), gprIndex("RSI"), gprIndex("RDX"),
    gprIndex("RCX"), gprIndex("R8"), gprIndex("R9")
  };
  unsigned intReg = 0;
  unsigned fpReg = 0;
  Type* i64Ty = Type::getInt64Ty(ctx);
  Type* f64Ty = Type::getDoubleTy(ctx);
  auto* xmmVecTy = FixedVectorType::get(f64Ty, 2);

  for (Argument& arg : wrapper->args()) {
    Type* aTy = arg.getType();
    if (aTy->isFloatTy() || aTy->isDoubleTy()) {
      if (fpReg < 8) {
        unsigned fieldIdx = kXmmBase + fpReg;
        Value* xmmPtr = B.CreateStructGEP(stateTy, state, fieldIdx);
        Value* lane0 = &arg;
        if (aTy->isFloatTy()) {
          lane0 = B.CreateFPExt(&arg, f64Ty);
        }
        Value* vec = UndefValue::get(xmmVecTy);
        vec = B.CreateInsertElement(vec, lane0, B.getInt32(0));
        vec = B.CreateInsertElement(vec, ConstantFP::get(f64Ty, 0.0), B.getInt32(1));
        B.CreateStore(vec, xmmPtr);
      }
      fpReg++;
      continue;
    }
    if (intReg < 6) {
      int idx = gprOrder[intReg];
      if (idx >= 0) {
        Value* gprPtr = B.CreateStructGEP(stateTy, state, (unsigned)idx);
        Value* val = nullptr;
        if (aTy->isPointerTy()) {
          val = B.CreatePtrToInt(&arg, i64Ty);
        } else if (aTy->isIntegerTy()) {
          unsigned bits = cast<IntegerType>(aTy)->getBitWidth();
          if (bits < 64) val = B.CreateZExt(&arg, i64Ty);
          else if (bits == 64) val = &arg;
          else val = B.CreateTrunc(&arg, i64Ty);
        } else {
          val = B.CreateZExtOrBitCast(&arg, i64Ty);
        }
        B.CreateStore(val, gprPtr);
      }
    }
    intReg++;
  }

  // 6. Call lifted function (explicit bitcast matches spec wording exactly).
  Value* stateForCall = B.CreateBitCast(state, PointerType::getUnqual(ctx));
  B.CreateCall(liftedF, {stateForCall});

  // 7. Return or Unreachable.
  if (wrapperNoReturn) {
    B.CreateUnreachable();
    return;
  }
  Type* retTy = wrapper->getReturnType();
  if (retTy->isVoidTy()) {
    B.CreateRetVoid();
    return;
  }
  if (retTy->isFloatTy() || retTy->isDoubleTy()) {
    Value* xmm0Ptr = B.CreateStructGEP(stateTy, state, kXmmBase + 0);
    Value* xmm0 = B.CreateLoad(xmmVecTy, xmm0Ptr);
    Value* lane0 = B.CreateExtractElement(xmm0, B.getInt32(0));
    if (retTy->isFloatTy()) {
      lane0 = B.CreateFPTrunc(lane0, retTy);
    }
    B.CreateRet(lane0);
    return;
  }
  Value* raxPtr = B.CreateStructGEP(stateTy, state, 0);
  Value* ret64 = B.CreateLoad(i64Ty, raxPtr);
  if (retTy->isPointerTy()) {
    B.CreateRet(B.CreateIntToPtr(ret64, retTy));
    return;
  }
  if (retTy->isIntegerTy()) {
    unsigned bits = cast<IntegerType>(retTy)->getBitWidth();
    if (bits < 64) {
      B.CreateRet(B.CreateTrunc(ret64, retTy));
      return;
    }
    if (bits == 64) {
      B.CreateRet(ret64);
      return;
    }
  }
  B.CreateRet(UndefValue::get(retTy));
}

int main(int argc, char** argv) {
  bool printMode = false;
  std::vector<std::string> positionalArgs;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--print") {
      printMode = true;
    } else {
      positionalArgs.push_back(a);
    }
  }

  std::string inputPath;
  std::string bcOut  = printMode ? "lifted_module.ll"  : "lifted_module.bc";
  std::string stateOut = printMode ? "step10_state.json" : "step10_state.pb";
  std::string irOut;

  if (positionalArgs.size() > 0) inputPath = positionalArgs[0];
  if (positionalArgs.size() > 1) {
    std::string a2 = positionalArgs[1];
    if (endsWith(a2, ".ll") && !printMode) {
      irOut = a2;
    } else {
      bcOut = a2;
      if (positionalArgs.size() > 2) stateOut = positionalArgs[2];
      if (positionalArgs.size() > 3) irOut = positionalArgs[3];
    }
  }

  lifted_ast::Program proto;
  if (!inputPath.empty()) {
    std::ifstream in(inputPath, std::ios::binary);
    if (!proto.ParseFromIstream(&in)) {
      std::cerr << "Protobuf parse failed\n";
      return 1;
    }
  } else {
    if (!proto.ParseFromIstream(&std::cin)) {
      std::cerr << "Protobuf parse failed\n";
      return 1;
    }
  }

  LLVMContext ctx;
  auto M = std::make_unique<Module>("lifted", ctx);
  M->setTargetTriple("x86_64-unknown-linux-gnu");
  M->setDataLayout("e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128");

  StructType* stateTy = createStateType(ctx);
  PointerType* statePtrTy = PointerType::getUnqual(ctx);

  std::unordered_map<std::string, GlobalValue*> symMap;

  // 1) Emit globals, constants, and external declarations.
  for (const auto& it : proto.symbol_table()) {
    const std::string& name = it.first;
    const lifted_ast::SymbolEntry& entry = it.second;

    if (name == "llvm.global_ctors" || name == "llvm.global_dtors") continue;

    const std::string kind = entry.has_kind() ? entry.kind() : "";

    if (kind == "function") {
      if (entry.has_is_external() && entry.is_external()) {
        FunctionType* ft = parseFunctionType(ctx, entry.llvm_type());
        auto* f = Function::Create(ft, GlobalValue::ExternalLinkage, name, M.get());

        if (name == "exit") f->addFnAttr(Attribute::NoReturn);

        symMap[name] = f;
      }
      continue;
    }

    if (kind == "data" || kind == "constant") {
      Type* gTy = parseLLVMType(ctx, entry.llvm_type());

      if (entry.has_is_external() && entry.is_external()) {
        auto* gv = new GlobalVariable(*M, gTy,
                                      /*isConstant=*/false,
                                      GlobalValue::ExternalLinkage,
                                      /*Initializer=*/nullptr,
                                      name);
        symMap[name] = gv;
        continue;
      }

      Constant* init = buildInitializer(ctx, gTy, entry);

      bool isBss = entry.has_section() && (entry.section().find("bss") != std::string::npos);
      bool isConst = (kind == "constant") ? true : !isBss;

      auto linkage = resolveLinkage(entry);

      auto* gv = new GlobalVariable(*M, gTy, isConst, linkage, init, name);
      symMap[name] = gv;
      continue;
    }
  }

  // 2) Lifted function declarations + boundary ABI wrappers.
  for (const auto& sec : proto.sections()) {
    for (const auto& ch : sec.children()) {
      if (!ch.has_function()) continue;
      const lifted_ast::Function& fproto = ch.function();
      std::string entry = fproto.entry_label();

      const auto& lsig = fproto.lifted_signature();
      Type* liftedRetTy = parseLLVMType(ctx, lsig.return_type());
      FunctionType* liftedFT = FunctionType::get(liftedRetTy, {statePtrTy}, false);
      std::string liftedName = entry + "_lifted";

      // Declare lifted function (Internal Linkage).
      Function* liftedF = Function::Create(liftedFT, GlobalValue::InternalLinkage, liftedName, M.get());
      applyLiftedAttributes(liftedF, lsig);
      // Create stub body to satisfy verifyModule for InternalLinkage.
      createStubBody(liftedF);
      symMap[liftedName] = liftedF;

      if (fproto.is_boundary()) {
        FunctionType* wrapperFT = nullptr;
        if (fproto.has_external_abi_signature()) {
          wrapperFT = parseFunctionType(ctx, fproto.external_abi_signature());
        } else {
          wrapperFT = FunctionType::get(parseLLVMType(ctx, fproto.return_type()), {}, false);
        }

        // Linkage exactly matching the original symbol binding.
        GlobalValue::LinkageTypes link = GlobalValue::ExternalLinkage;
        auto sit = proto.symbol_table().find(entry);
        if (sit != proto.symbol_table().end()) {
          link = resolveLinkage(sit->second);
        }

        Function* wrapper = Function::Create(wrapperFT, link, entry, M.get());
        symMap[entry] = wrapper;

        bool wrapperNoReturn = false;
        for (const auto& a : lsig.attributes()) {
          if (a == "noreturn") wrapperNoReturn = true;
        }
        if (wrapperNoReturn) wrapper->addFnAttr(Attribute::NoReturn);

        // Synthesize complete simple marshaling body.
        synthesizeBoundaryWrapperBody(ctx, stateTy, wrapper, liftedF, wrapperNoReturn);
      }
    }
  }

  // 3) Special creation of llvm.global_ctors / llvm.global_dtors.
  for (const std::string& name : {"llvm.global_ctors", "llvm.global_dtors"}) {
    auto it = proto.symbol_table().find(name);
    if (it == proto.symbol_table().end()) continue;

    StructType* ctorStruct = StructType::get(ctx, {
      Type::getInt32Ty(ctx),
      PointerType::getUnqual(ctx),
      PointerType::getUnqual(ctx)
    });

    uint64_t n = 1;
    std::string tyStr = it->second.llvm_type();
    if (!tyStr.empty() && tyStr.front() == '[') {
      auto xPos = tyStr.find(" x ");
      if (xPos != std::string::npos) {
        auto nStr = trim(tyStr.substr(1, xPos - 1));
        n = std::stoull(nStr);
      }
    }

    ArrayType* arrTy = ArrayType::get(ctorStruct, n);
    std::string target = (name == "llvm.global_ctors") ? "constructor_stub" : "destructor_stub";

    Constant* initializer = ConstantAggregateZero::get(arrTy);
    if (symMap.count(target) && n >= 1) {
      Function* wrapperF = cast<Function>(symMap[target]);
      Constant* prio = ConstantInt::get(Type::getInt32Ty(ctx), 65535);
      Constant* fptr = ConstantExpr::getBitCast(wrapperF, PointerType::getUnqual(ctx));
      Constant* nullp = ConstantPointerNull::get(PointerType::getUnqual(ctx));
      Constant* structVal = ConstantStruct::get(ctorStruct, {prio, fptr, nullp});

      SmallVector<Constant*, 8> elts;
      elts.reserve((size_t)n);
      elts.push_back(structVal);
      while (elts.size() < (size_t)n) elts.push_back(ConstantAggregateZero::get(ctorStruct));

      initializer = ConstantArray::get(arrTy, elts);
    }

    auto* gv = new GlobalVariable(*M, arrTy, false, GlobalValue::AppendingLinkage, initializer, name);
    symMap[name] = gv;
  }

  // 4) PIC relocation named metadata.
  auto* picRelocMD = M->getOrInsertNamedMetadata("pic_relocations");
  for (const auto& sym : proto.symbol_table()) {
    const std::string& symName = sym.first;
    const lifted_ast::SymbolEntry& symEntry = sym.second;

    std::string llvmTargetName = symName;
    if (auto itv = symMap.find(symName); itv != symMap.end()) {
      llvmTargetName = itv->second->getName().str();
    }

    for (const auto& reloc : symEntry.relocations()) {
      SmallVector<Metadata*, 4> fields;
      fields.push_back(MDString::get(ctx, reloc.has_type() ? reloc.type() : ""));
      fields.push_back(MDString::get(ctx, llvmTargetName));
      fields.push_back(MDString::get(ctx, reloc.has_instruction() ? reloc.instruction() : ""));
      fields.push_back(ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt1Ty(ctx), reloc.has_pic() && reloc.pic())));
      picRelocMD->addOperand(MDNode::get(ctx, fields));
    }
  }

  // 5) Module Validation.
  if (verifyModule(*M, &errs())) {
    errs() << "LLVM module verification FAILED\n";
    return 1;
  }

  // 6) Populate stable LLVM name mappings (liftedRef / wrapperRef) and serialize augmented protobuf state.
  {
    auto* st = proto.mutable_symbol_table();
    for (auto& kv : *st) {
      const std::string& symName = kv.first;
      lifted_ast::SymbolEntry& entry = kv.second;

      auto itDirect = symMap.find(symName);
      auto itLifted = symMap.find(symName + "_lifted");

      // lifted_ref for ALL functions with a lifted version — actual LLVM name.
      if (itLifted != symMap.end()) {
        entry.set_lifted_ref(itLifted->second->getName().str());
      }

      // wrapper_ref for boundary functions only — actual LLVM wrapper name.
      if (entry.has_is_boundary() && entry.is_boundary() && itDirect != symMap.end()) {
        entry.set_wrapper_ref(itDirect->second->getName().str());
      }
    }

    if (printMode) {
      std::string jsonStr;
      google::protobuf::util::JsonPrintOptions opts;
      opts.add_whitespace = true;
      auto status = google::protobuf::util::MessageToJsonString(proto, &jsonStr, opts);
      if (!status.ok()) {
        std::cerr << "Failed to convert protobuf to JSON: " << status.ToString() << "\n";
        return 1;
      }
      std::ofstream out(stateOut, std::ios::trunc);
      if (!out) {
        std::cerr << "Cannot open state output: " << stateOut << "\n";
        return 1;
      }
      out << jsonStr;
    } else {
      std::ofstream out(stateOut, std::ios::binary | std::ios::trunc);
      if (!out) {
        std::cerr << "Cannot open state output: " << stateOut << "\n";
        return 1;
      }
      if (!proto.SerializeToOstream(&out)) {
        std::cerr << "Failed to serialize augmented protobuf state\n";
        return 1;
      }
    }
  }

  // 7) Write verified module.
  {
    std::error_code EC;
    raw_fd_ostream os(bcOut, EC, sys::fs::OF_None);
    if (EC) {
      std::cerr << "Cannot open module output: " << bcOut << "\n";
      return 1;
    }
    if (printMode) {
      M->print(os, nullptr);
    } else {
      WriteBitcodeToFile(*M, os);
    }
    os.flush();
  }

  // Optional: emit textual IR (debug convenience).
  if (!irOut.empty()) {
    std::error_code EC;
    raw_fd_ostream irOS(irOut, EC, sys::fs::OF_None);
    if (EC) {
      std::cerr << "Cannot open IR output: " << irOut << "\n";
      return 1;
    }
    M->print(irOS, nullptr);
    irOS.flush();
  }

  return 0;
}
