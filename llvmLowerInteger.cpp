/*
llvmLowerInteger.cpp  (Implementation of Step 11)
Leverages llvmLiftShared.hpp for shared definitions, RValue/LValue decoding, and Memory translation.

Supported Instructions:
Data movement: MOV, LEA, MOVZX, MOVSX, XCHG, PUSH, POP, LEAVE, CDQ
Arithmetic / Logic: ADD, SUB, XOR, AND, OR, DIV, INC, IMUL, IDIV
Comparison / Flags: CMP, TEST, CMOVE, SETcc
Shifts: SHL, SHR, SAR
*/

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include "ast.pb.h"
#include <google/protobuf/util/json_util.h>

#include "llvmLiftShared.hpp"

using namespace llvm;
using namespace llvm_lift;

// ----------------------------- local flags utilities -----------------------------

static Value* computePF(IRBuilder<>& B, Value* resIntN) {
  Value* lo8 = B.CreateTrunc(resIntN, B.getInt8Ty(), "pf.lo8");
  Function* ctpop = Intrinsic::getDeclaration(B.GetInsertBlock()->getModule(), Intrinsic::ctpop, {B.getInt8Ty()});
  Value* pop = B.CreateCall(ctpop, {lo8}, "pf.pop");
  Value* lsb = B.CreateAnd(pop, B.getInt8(1), "pf.lsb");
  Value* odd = B.CreateICmpEQ(lsb, B.getInt8(1), "pf.odd");
  return B.CreateNot(odd, "pf");
}

static void updateFlagsLogic(FnLowerCtx& LC, IRBuilder<>& B, Value* res) {
  storeFlag(LC, B, CF, B.getFalse());
  storeFlag(LC, B, OF, B.getFalse());
  storeFlag(LC, B, AF, B.getFalse());
  storeFlag(LC, B, ZF, B.CreateICmpEQ(res, ConstantInt::get(res->getType(), 0), "zf"));
  storeFlag(LC, B, SF, B.CreateICmpSLT(res, ConstantInt::get(res->getType(), 0), "sf"));
  storeFlag(LC, B, PF, computePF(B, res));
}

static void updateFlagsAddSubCommon(FnLowerCtx& LC, IRBuilder<>& B, Value* a, Value* b, Value* res, Value* cf, Value* of) {
  storeFlag(LC, B, ZF, B.CreateICmpEQ(res, ConstantInt::get(res->getType(), 0), "zf"));
  storeFlag(LC, B, SF, B.CreateICmpSLT(res, ConstantInt::get(res->getType(), 0), "sf"));
  storeFlag(LC, B, PF, computePF(B, res));
  Value* x = B.CreateXor(a, b, "af.x1");
  x = B.CreateXor(x, res, "af.x2");
  Value* mask = ConstantInt::get(res->getType(), 0x10);
  storeFlag(LC, B, AF, B.CreateICmpNE(B.CreateAnd(x, mask), ConstantInt::get(res->getType(), 0), "af"));
  storeFlag(LC, B, CF, cf);
  storeFlag(LC, B, OF, of);
}

static void updateFlagsAdd(FnLowerCtx& LC, IRBuilder<>& B, Value* a, Value* b, Value* res) {
  Type* ty = res->getType();
  Function* uadd = Intrinsic::getDeclaration(&LC.M, Intrinsic::uadd_with_overflow, {ty});
  Function* sadd = Intrinsic::getDeclaration(&LC.M, Intrinsic::sadd_with_overflow, {ty});
  Value* u = B.CreateCall(uadd, {a, b}, "uadd.ov");
  Value* s = B.CreateCall(sadd, {a, b}, "sadd.ov");
  updateFlagsAddSubCommon(LC, B, a, b, res, B.CreateExtractValue(u, 1, "cf"), B.CreateExtractValue(s, 1, "of"));
}

static void updateFlagsSub(FnLowerCtx& LC, IRBuilder<>& B, Value* a, Value* b, Value* res) {
  Type* ty = res->getType();
  Function* usub = Intrinsic::getDeclaration(&LC.M, Intrinsic::usub_with_overflow, {ty});
  Function* ssub = Intrinsic::getDeclaration(&LC.M, Intrinsic::ssub_with_overflow, {ty});
  Value* u = B.CreateCall(usub, {a, b}, "usub.ov");
  Value* s = B.CreateCall(ssub, {a, b}, "ssub.ov");
  updateFlagsAddSubCommon(LC, B, a, b, res, B.CreateExtractValue(u, 1, "cf"), B.CreateExtractValue(s, 1, "of"));
}

// ----------------------------- AST helper utilities ----------------------------

static bool isTerminatorOpcode(const std::string& opcUpper) {
  static const std::set<std::string> terms = {
    "RET","JMP","JE","JNE","JL","JLE","JG","JGE","JA","JAE","JB","JBE",
    "JO","JNO","JS","JNS","JP","JNP","JC","JNC",
    "LOOP","LOOPE","LOOPNE","IRET","SYSRET"
  };
  return terms.count(opcUpper) != 0;
}

static bool isFloatOpcode(const std::string& opcUpper) {
  static const std::set<std::string> fp = {
    "MOVSS","ADDSS","SUBSS","MULSS","DIVSS",
    "MOVSD","ADDSD","SUBSD","MULSD","DIVSD",
    "CVTTSS2SI","CVTSS2SI","CVTSI2SS",
    "CVTTSD2SI","CVTSD2SI","CVTSI2SD"
  };
  return fp.count(opcUpper) != 0;
}

static unsigned regWidthBits(const std::string& reg) {
  RegInfo ri = decodeReg(reg);
  if (!ri.isValid || ri.isXmm) return 0;
  return ri.bitWidth;
}

static Type* chooseOpIntType(FnLowerCtx& LC, const lifted_ast::Instruction& insn) {
  if (insn.has_op_refinement()) {
    std::string r = toUpper(insn.op_refinement());
    if (r == "I8" || r.rfind("I8", 0) == 0 || r == "CHAR") return Type::getInt8Ty(LC.C);
    if (r == "I16" || r.rfind("I16", 0) == 0) return Type::getInt16Ty(LC.C);
    if (r == "I32" || r.rfind("I32", 0) == 0) return Type::getInt32Ty(LC.C);
    if (r == "I64" || r.rfind("I64", 0) == 0 || r == "PTR") return Type::getInt64Ty(LC.C);
  }
  return Type::getInt64Ty(LC.C);
}

// ----------------------------- Lowering Implementations -------------------------

static void lowerMOV(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 2) return;
  const auto& dst = insn.operands(0);
  const auto& src = insn.operands(1);
  Type* ty = nullptr;

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

  Value* v = resolveRValue(LC, B, src, ty, insn.id());

  if (dst.has_register_()) {
    RegInfo dri = decodeReg(dst.register_());
    writeGprSubreg(LC, B, dri, v);
    recordMapping(LC, insn, v);
  } else if (dst.has_memory()) {
    storeToMem(LC, B, dst, v, ty, insn.id());
    recordMapping(LC, insn, v);
  }
}

static void lowerLEA(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 2) return;
  const auto& dst = insn.operands(0);
  const auto& src = insn.operands(1);
  if (!dst.has_register_()) return;
  RegInfo dri = decodeReg(dst.register_());
  if (!dri.isValid || dri.isXmm) return;

  Value* addrI64 = nullptr;
  if (src.has_memory()) {
    MemAddr a = resolveMemAddress(LC, B, src, 1, insn.id());
    Value* pi64 = B.CreatePtrToInt(a.ptr, B.getInt64Ty(), "lea.ptrtoint");
    if (isa<Instruction>(pi64)) attachAstInstrId(cast<Instruction>(pi64), insn.id(), LC.C);
    addrI64 = pi64;
  } else if (src.has_symbol_ref()) {
    Value* p = symbolAddressAsPtr(LC, B, src.symbol_ref(), insn.id());
    Value* pi64 = B.CreatePtrToInt(p, B.getInt64Ty(), "lea.sym");
    if (isa<Instruction>(pi64)) attachAstInstrId(cast<Instruction>(pi64), insn.id(), LC.C);

    int64_t addend = 0;
    if (src.has_integer() && src.integer().has_value()) {
      if (auto iOpt = valueAsInt64(src.integer().value())) addend = *iOpt;
    }
    if (addend != 0) pi64 = B.CreateAdd(pi64, B.getInt64(addend), "lea.sym.add");
    addrI64 = pi64;
  } else if (src.has_expression()) {
    addrI64 = evalExprToI64(LC, B, src.expression(), insn.id());
  } else {
    addrI64 = UndefValue::get(B.getInt64Ty());
  }

  Type* dstTy = intTy(LC.C, dri.bitWidth);
  Value* out = truncOrZext(B, addrI64, dstTy);
  writeGprSubreg(LC, B, dri, out);
  recordMapping(LC, insn, out);
}

static void lowerALU2(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Instruction& insn, const std::string& opcUpper) {
  if (insn.operands_size() < 2) return;
  const auto& dst = insn.operands(0);
  const auto& src = insn.operands(1);
  Type* ty = nullptr;

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

  Value* a = nullptr;
  if (dst.has_register_()) {
    a = truncOrZext(B, readGprSubreg(LC, B, decodeReg(dst.register_())), ty);
  } else if (dst.has_memory()) {
    a = loadFromMem(LC, B, dst, ty, insn.id());
  } else { return; }

  Value* b = resolveRValue(LC, B, src, ty, insn.id());
  Value* res = nullptr;
  bool isLogic = false;

  if (opcUpper == "ADD") { res = B.CreateAdd(a, b, "add"); updateFlagsAdd(LC, B, a, b, res); }
  else if (opcUpper == "SUB") { res = B.CreateSub(a, b, "sub"); updateFlagsSub(LC, B, a, b, res); }
  else if (opcUpper == "XOR") { res = B.CreateXor(a, b, "xor"); isLogic = true; }
  else if (opcUpper == "AND") { res = B.CreateAnd(a, b, "and"); isLogic = true; }
  else if (opcUpper == "OR") { res = B.CreateOr(a, b, "or"); isLogic = true; }
  else return;

  if (isLogic) updateFlagsLogic(LC, B, res);

  if (dst.has_register_()) writeGprSubreg(LC, B, decodeReg(dst.register_()), res);
  else if (dst.has_memory()) storeToMem(LC, B, dst, res, ty, insn.id());
  recordMapping(LC, insn, res);
}

static void lowerCMP(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 2) return;
  const auto& aOp = insn.operands(0);
  const auto& bOp = insn.operands(1);

  Type* ty = nullptr;
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

  Value* a = resolveRValue(LC, B, aOp, ty, insn.id());
  Value* b = resolveRValue(LC, B, bOp, ty, insn.id());
  Value* res = B.CreateSub(a, b, "cmp.sub");
  updateFlagsSub(LC, B, a, b, res);
  recordMapping(LC, insn, res);
}

static void lowerTEST(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 2) return;
  const auto& aOp = insn.operands(0);
  const auto& bOp = insn.operands(1);

  Type* ty = nullptr;
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

  Value* a = resolveRValue(LC, B, aOp, ty, insn.id());
  Value* b = resolveRValue(LC, B, bOp, ty, insn.id());
  Value* res = B.CreateAnd(a, b, "test.and");
  updateFlagsLogic(LC, B, res);
  recordMapping(LC, insn, res);
}

static void lowerMOVZX_MOVSX(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Instruction& insn, bool isSignExtend) {
  if (insn.operands_size() < 2) return;
  const auto& dst = insn.operands(0);
  const auto& src = insn.operands(1);

  if (!dst.has_register_()) return;
  RegInfo dri = decodeReg(dst.register_());
  if (!dri.isValid || dri.isXmm) return;

  Type* dstTy = intTy(LC.C, dri.bitWidth);
  Type* srcTy = nullptr;
  if (src.has_register_()) {
    unsigned sw = regWidthBits(src.register_());
    srcTy = intTy(LC.C, sw == 0 ? 8 : sw);
  } else if (src.has_memory()) {
    unsigned sz = memSizeBytesFromOperand(src);
    srcTy = intTy(LC.C, sz == 0 ? 8 : (sz * 8));
  } else {
    srcTy = dstTy;
  }

  Value* v = resolveRValue(LC, B, src, srcTy, insn.id());
  Value* ext = isSignExtend ? truncOrSext(B, v, dstTy) : truncOrZext(B, v, dstTy);
  writeGprSubreg(LC, B, dri, ext);
  recordMapping(LC, insn, ext);
}

static void lowerSETcc(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Instruction& insn, const std::string& opcUpper) {
  if (insn.operands_size() < 1) return;
  const auto& dst = insn.operands(0);

  Value* cond = nullptr;
  if (opcUpper == "SETC" || opcUpper == "SETB") cond = loadFlag(LC, B, CF);
  else if (opcUpper == "SETNE" || opcUpper == "SETNZ") cond = B.CreateNot(loadFlag(LC, B, ZF), "setnz");
  else if (opcUpper == "SETE" || opcUpper == "SETZ") cond = loadFlag(LC, B, ZF);
  else if (opcUpper == "SETPE" || opcUpper == "SETP") cond = loadFlag(LC, B, PF);
  else cond = B.getFalse();

  Value* byteV = B.CreateZExt(cond, B.getInt8Ty(), "setcc.i8");
  if (dst.has_register_()) writeGprSubreg(LC, B, decodeReg(dst.register_()), byteV);
  else if (dst.has_memory()) storeToMem(LC, B, dst, byteV, B.getInt8Ty(), insn.id());
  recordMapping(LC, insn, byteV);
}

static void lowerCMOVE(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 2) return;
  const auto& dst = insn.operands(0);
  const auto& src = insn.operands(1);
  if (!dst.has_register_()) return;
  RegInfo dri = decodeReg(dst.register_());
  if (!dri.isValid || dri.isXmm) return;

  Type* ty = intTy(LC.C, dri.bitWidth);
  Value* oldv = truncOrZext(B, readGprSubreg(LC, B, dri), ty);
  Value* newv = resolveRValue(LC, B, src, ty, insn.id());

  Value* res = B.CreateSelect(loadFlag(LC, B, ZF), newv, oldv, "cmove.sel");
  writeGprSubreg(LC, B, dri, res);
  recordMapping(LC, insn, res);
}

static void lowerINC(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 1) return;
  const auto& dst = insn.operands(0);

  Type* ty = nullptr;
  if (dst.has_register_()) {
    unsigned w = regWidthBits(dst.register_());
    ty = intTy(LC.C, w == 0 ? 64 : w);
  } else if (dst.has_memory()) {
    unsigned sz = memSizeBytesFromOperand(dst);
    ty = sz == 0 ? chooseOpIntType(LC, insn) : intTy(LC.C, sz * 8);
  } else return;
  if (!ty || !ty->isIntegerTy()) ty = B.getInt64Ty();

  Value* old = dst.has_register_() ? truncOrZext(B, readGprSubreg(LC, B, decodeReg(dst.register_())), ty) : loadFromMem(LC, B, dst, ty, insn.id());
  Value* one = ConstantInt::get(ty, 1);
  Value* res = B.CreateAdd(old, one, "inc");

  Function* sadd = Intrinsic::getDeclaration(&LC.M, Intrinsic::sadd_with_overflow, {ty});
  Value* s = B.CreateCall(sadd, {old, one}, "inc.ov");
  storeFlag(LC, B, OF, B.CreateExtractValue(s, 1, "of"));
  storeFlag(LC, B, ZF, B.CreateICmpEQ(res, ConstantInt::get(ty, 0), "zf"));
  storeFlag(LC, B, SF, B.CreateICmpSLT(res, ConstantInt::get(ty, 0), "sf"));
  storeFlag(LC, B, PF, computePF(B, res));
  Value* x = B.CreateXor(old, one); x = B.CreateXor(x, res);
  storeFlag(LC, B, AF, B.CreateICmpNE(B.CreateAnd(x, ConstantInt::get(ty, 0x10)), ConstantInt::get(ty, 0)));

  if (dst.has_register_()) writeGprSubreg(LC, B, decodeReg(dst.register_()), res);
  else storeToMem(LC, B, dst, res, ty, insn.id());
  recordMapping(LC, insn, res);
}

static void lowerSHL(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 2) return;
  const auto& dst = insn.operands(0);
  const auto& cnt = insn.operands(1);

  Type* ty = nullptr;
  if (dst.has_register_()) {
    unsigned w = regWidthBits(dst.register_());
    ty = intTy(LC.C, w == 0 ? 64 : w);
  } else if (dst.has_memory()) {
    unsigned sz = memSizeBytesFromOperand(dst);
    ty = sz == 0 ? chooseOpIntType(LC, insn) : intTy(LC.C, sz * 8);
  } else return;
  if (!ty || !ty->isIntegerTy()) ty = B.getInt64Ty();

  Value* old = dst.has_register_() ? truncOrZext(B, readGprSubreg(LC, B, decodeReg(dst.register_())), ty) : loadFromMem(LC, B, dst, ty, insn.id());

  Value* count = nullptr;
  if (cnt.has_integer()) count = ConstantInt::get(ty, (uint64_t)valueAsInt64(cnt.integer().value()).value_or(0));
  else if (cnt.has_register_()) count = truncOrZext(B, readGprSubreg(LC, B, decodeReg(cnt.register_())), ty);
  else count = ConstantInt::get(ty, 0);

  unsigned w = ty->getIntegerBitWidth();
  Value* mask = ConstantInt::get(ty, (uint64_t)(w - 1));
  Value* masked = B.CreateAnd(count, mask, "shl.mask");
  Value* res = B.CreateShl(old, masked, "shl");

  Value* shiftAmtI64 = truncOrZext(B, masked, B.getInt64Ty());
  Value* pos = B.CreateSub(B.getInt64(w), shiftAmtI64, "shl.pos");
  Value* cfBit = B.CreateLShr(old, truncOrZext(B, pos, ty));
  storeFlag(LC, B, CF, B.CreateTrunc(cfBit, B.getInt1Ty(), "cf"));
  storeFlag(LC, B, OF, B.getFalse());
  storeFlag(LC, B, ZF, B.CreateICmpEQ(res, ConstantInt::get(ty, 0), "zf"));
  storeFlag(LC, B, SF, B.CreateICmpSLT(res, ConstantInt::get(ty, 0), "sf"));
  storeFlag(LC, B, PF, computePF(B, res));

  if (dst.has_register_()) writeGprSubreg(LC, B, decodeReg(dst.register_()), res);
  else storeToMem(LC, B, dst, res, ty, insn.id());
  recordMapping(LC, insn, res);
}

static void lowerSHR(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 2) return;
  const auto& dst = insn.operands(0);
  const auto& cnt = insn.operands(1);

  Type* ty = nullptr;
  if (dst.has_register_()) {
    unsigned w = regWidthBits(dst.register_());
    ty = intTy(LC.C, w == 0 ? 64 : w);
  } else if (dst.has_memory()) {
    unsigned sz = memSizeBytesFromOperand(dst);
    ty = sz == 0 ? chooseOpIntType(LC, insn) : intTy(LC.C, sz * 8);
  } else return;
  if (!ty || !ty->isIntegerTy()) ty = B.getInt64Ty();

  Value* old = dst.has_register_() ? truncOrZext(B, readGprSubreg(LC, B, decodeReg(dst.register_())), ty) : loadFromMem(LC, B, dst, ty, insn.id());

  Value* count = nullptr;
  if (cnt.has_integer()) count = ConstantInt::get(ty, (uint64_t)valueAsInt64(cnt.integer().value()).value_or(0));
  else if (cnt.has_register_()) count = truncOrZext(B, readGprSubreg(LC, B, decodeReg(cnt.register_())), ty);
  else count = ConstantInt::get(ty, 0);

  unsigned w = ty->getIntegerBitWidth();
  Value* mask = ConstantInt::get(ty, (uint64_t)(w - 1));
  Value* masked = B.CreateAnd(count, mask, "shr.mask");
  Value* res = B.CreateLShr(old, masked, "shr");

  Value* doUpdate = B.CreateICmpNE(masked, ConstantInt::get(ty, 0));
  Value* cfShiftAmt = B.CreateSub(masked, ConstantInt::get(ty, 1));
  Value* tmp = B.CreateLShr(old, cfShiftAmt);
  Value* lastBit = B.CreateAnd(tmp, ConstantInt::get(ty, 1));
  Value* newCf = B.CreateTrunc(lastBit, B.getInt1Ty());
  storeFlag(LC, B, CF, B.CreateSelect(doUpdate, newCf, loadFlag(LC, B, CF)));

  Value* isOne = B.CreateICmpEQ(masked, ConstantInt::get(ty, 1));
  Value* msb = B.CreateLShr(old, ConstantInt::get(ty, w - 1));
  Value* ofOne = B.CreateTrunc(B.CreateAnd(msb, ConstantInt::get(ty, 1)), B.getInt1Ty());
  storeFlag(LC, B, OF, B.CreateSelect(isOne, ofOne, loadFlag(LC, B, OF)));

  storeFlag(LC, B, ZF, B.CreateICmpEQ(res, ConstantInt::get(ty, 0), "zf"));
  storeFlag(LC, B, SF, B.CreateICmpSLT(res, ConstantInt::get(ty, 0), "sf"));
  storeFlag(LC, B, PF, computePF(B, res));

  if (dst.has_register_()) writeGprSubreg(LC, B, decodeReg(dst.register_()), res);
  else storeToMem(LC, B, dst, res, ty, insn.id());
  recordMapping(LC, insn, res);
}

static void lowerSAR(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 2) return;
  const auto& dst = insn.operands(0);
  const auto& cnt = insn.operands(1);

  Type* ty = nullptr;
  if (dst.has_register_()) {
    unsigned w = regWidthBits(dst.register_());
    ty = intTy(LC.C, w == 0 ? 64 : w);
  } else if (dst.has_memory()) {
    unsigned sz = memSizeBytesFromOperand(dst);
    ty = sz == 0 ? chooseOpIntType(LC, insn) : intTy(LC.C, sz * 8);
  } else return;
  if (!ty || !ty->isIntegerTy()) ty = B.getInt64Ty();

  Value* old = dst.has_register_() ? truncOrZext(B, readGprSubreg(LC, B, decodeReg(dst.register_())), ty) : loadFromMem(LC, B, dst, ty, insn.id());

  Value* count = nullptr;
  if (cnt.has_integer()) count = ConstantInt::get(ty, (uint64_t)valueAsInt64(cnt.integer().value()).value_or(0));
  else if (cnt.has_register_()) count = truncOrZext(B, readGprSubreg(LC, B, decodeReg(cnt.register_())), ty);
  else count = ConstantInt::get(ty, 0);

  unsigned w = ty->getIntegerBitWidth();
  Value* mask = ConstantInt::get(ty, (uint64_t)(w - 1));
  Value* masked = B.CreateAnd(count, mask, "sar.mask");
  Value* res = B.CreateAShr(old, masked, "sar");

  Value* doUpdate = B.CreateICmpNE(masked, ConstantInt::get(ty, 0));
  Value* cfShiftAmt = B.CreateSub(masked, ConstantInt::get(ty, 1));
  Value* tmp = B.CreateLShr(old, cfShiftAmt);
  Value* lastBit = B.CreateAnd(tmp, ConstantInt::get(ty, 1));
  Value* newCf = B.CreateTrunc(lastBit, B.getInt1Ty());
  storeFlag(LC, B, CF, B.CreateSelect(doUpdate, newCf, loadFlag(LC, B, CF)));

  Value* isOne = B.CreateICmpEQ(masked, ConstantInt::get(ty, 1));
  storeFlag(LC, B, OF, B.CreateSelect(isOne, B.getFalse(), loadFlag(LC, B, OF)));

  storeFlag(LC, B, ZF, B.CreateICmpEQ(res, ConstantInt::get(ty, 0), "zf"));
  storeFlag(LC, B, SF, B.CreateICmpSLT(res, ConstantInt::get(ty, 0), "sf"));
  storeFlag(LC, B, PF, computePF(B, res));

  if (dst.has_register_()) writeGprSubreg(LC, B, decodeReg(dst.register_()), res);
  else storeToMem(LC, B, dst, res, ty, insn.id());
  recordMapping(LC, insn, res);
}

static void lowerXCHG(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 2) return;
  const auto& op0 = insn.operands(0);
  const auto& op1 = insn.operands(1);

  const lifted_ast::Operand* memOp = nullptr;
  const lifted_ast::Operand* regOp = nullptr;
  if (op0.has_memory() && op1.has_register_()) { memOp = &op0; regOp = &op1; }
  else if (op1.has_memory() && op0.has_register_()) { memOp = &op1; regOp = &op0; }
  else return;

  RegInfo rri = decodeReg(regOp->register_());
  if (!rri.isValid || rri.isXmm) return;
  Type* ty = intTy(LC.C, rri.bitWidth);
  Value* regVal = truncOrZext(B, readGprSubreg(LC, B, rri), ty);

  unsigned accessSize = rri.bitWidth / 8;
  if (accessSize == 0) accessSize = 1;
  MemAddr ma = resolveMemAddress(LC, B, *memOp, accessSize, insn.id());

  LoadInst* oldMem = B.CreateLoad(ty, ma.ptr, "xchg.ld");
  if (ma.align.value() > 1) oldMem->setAlignment(ma.align);
  if (ma.isSymbolic && !ma.symName.empty()) attachPicRelocations(oldMem, ma.symName, LC.C);
  attachAstInstrId(oldMem, insn.id(), LC.C);

  StoreInst* st = B.CreateStore(regVal, ma.ptr);
  if (ma.align.value() > 1) st->setAlignment(ma.align);
  if (ma.isSymbolic && !ma.symName.empty()) attachPicRelocations(st, ma.symName, LC.C);
  attachAstInstrId(st, insn.id(), LC.C);

  writeGprSubreg(LC, B, rri, oldMem);
  recordMapping(LC, insn, oldMem);
}

static void lowerXADD(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 2) return;
  const auto& dst = insn.operands(0);
  const auto& src = insn.operands(1);

  // Determine operand width from dst
  Type* ty = nullptr;
  if (dst.has_register_()) {
    unsigned w = regWidthBits(dst.register_());
    ty = intTy(LC.C, w == 0 ? 64 : w);
  } else if (dst.has_memory()) {
    unsigned sz = memSizeBytesFromOperand(dst);
    ty = sz == 0 ? chooseOpIntType(LC, insn) : intTy(LC.C, sz * 8);
  } else return;
  if (!ty || !ty->isIntegerTy()) ty = B.getInt64Ty();

  // Load destination (old value)
  Value* oldDst = nullptr;
  if (dst.has_register_()) {
    oldDst = truncOrZext(B, readGprSubreg(LC, B, decodeReg(dst.register_())), ty);
  } else if (dst.has_memory()) {
    oldDst = loadFromMem(LC, B, dst, ty, insn.id());
  } else return;

  // Load source value
  Value* srcVal = resolveRValue(LC, B, src, ty, insn.id());

  // Perform add
  Value* sum = B.CreateAdd(oldDst, srcVal, "xadd.sum");
  updateFlagsAdd(LC, B, oldDst, srcVal, sum);

  // Store sum back to destination
  if (dst.has_register_()) {
    writeGprSubreg(LC, B, decodeReg(dst.register_()), sum);
  } else if (dst.has_memory()) {
    storeToMem(LC, B, dst, sum, ty, insn.id());
  }

  // Write old destination value into source register
  // (XADD semantics: src <- old dst)
  if (src.has_register_()) {
    writeGprSubreg(LC, B, decodeReg(src.register_()), oldDst);
  }
  // Note: if src is memory, behavior is undefined per x86 (src must be reg)

  recordMapping(LC, insn, sum);
}

static void lowerPOP(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 1) return;
  const auto& dst = insn.operands(0);
  if (!dst.has_register_()) return;

  RegInfo dri = decodeReg(dst.register_());
  if (!dri.isValid || dri.isXmm) return;

  unsigned rspIdx = (unsigned)gprFieldIndex64("RSP");
  Value* rsp = loadGpr64(LC, B, rspIdx);
  Value* rspPtr = B.CreateIntToPtr(rsp, PointerType::getUnqual(LC.C), "pop.ptr");

  Type* ty = intTy(LC.C, dri.bitWidth);
  LoadInst* loaded = B.CreateLoad(ty, rspPtr, "pop.ld");
  loaded->setAlignment(Align(8));

  writeGprSubreg(LC, B, dri, loaded);

  Value* newRsp = B.CreateAdd(rsp, B.getInt64(8), "pop.rsp");
  storeGpr64(LC, B, rspIdx, newRsp);

  recordMapping(LC, insn, loaded);
}

static void lowerLEAVE(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Instruction& insn) {
  unsigned rspIdx = (unsigned)gprFieldIndex64("RSP");
  unsigned rbpIdx = (unsigned)gprFieldIndex64("RBP");

  Value* rbpVal = loadGpr64(LC, B, rbpIdx);
  storeGpr64(LC, B, rspIdx, rbpVal);

  Value* rspPtr = B.CreateIntToPtr(rbpVal, PointerType::getUnqual(LC.C), "leave.pop.ptr");
  LoadInst* loaded = B.CreateLoad(B.getInt64Ty(), rspPtr, "leave.pop.ld");
  loaded->setAlignment(Align(8));

  Value* newRsp = B.CreateAdd(rbpVal, B.getInt64(8), "leave.newrsp");
  storeGpr64(LC, B, rbpIdx, loaded);
  storeGpr64(LC, B, rspIdx, newRsp);

  recordMapping(LC, insn, loaded);
}

static void lowerPUSH(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 1) return;
  const auto& op = insn.operands(0);
  if (!op.has_register_()) return;
  RegInfo ri = decodeReg(op.register_());
  if (!ri.isValid || ri.isXmm) return;

  Value* val = readGprSubreg(LC, B, ri);
  Value* val64 = truncOrZext(B, val, B.getInt64Ty());

  unsigned rspIdx = (unsigned)gprFieldIndex64("RSP");
  Value* rsp = loadGpr64(LC, B, rspIdx);
  Value* newRsp = B.CreateSub(rsp, B.getInt64(8), "push.rsp");
  storeGpr64(LC, B, rspIdx, newRsp);

  Value* rspPtr = B.CreateIntToPtr(newRsp, PointerType::getUnqual(LC.C), "push.ptr");
  StoreInst* st = B.CreateStore(val64, rspPtr);
  st->setAlignment(Align(8));

  recordMapping(LC, insn, newRsp);
}

static void lowerIMUL(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Instruction& insn) {
  if (insn.operands_size() == 0) return;

  if (insn.operands_size() == 1) {
    const auto& srcOp = insn.operands(0);
    RegInfo raxRi = decodeReg("RAX");
    RegInfo rdxRi = decodeReg("RDX");

    Value* rax = readGprSubreg(LC, B, raxRi);
    Value* src = resolveRValue(LC, B, srcOp, B.getInt64Ty(), insn.id());

    Value* a128 = B.CreateSExt(rax, B.getInt128Ty(), "imul.a128");
    Value* b128 = B.CreateSExt(src, B.getInt128Ty(), "imul.b128");
    Value* prod = B.CreateMul(a128, b128, "imul.full");

    Value* low  = B.CreateTrunc(prod, B.getInt64Ty(), "imul.low");
    Value* high = B.CreateTrunc(B.CreateLShr(prod, 64), B.getInt64Ty(), "imul.high");

    writeGprSubreg(LC, B, raxRi, low);
    writeGprSubreg(LC, B, rdxRi, high);

    Value* signExt = B.CreateAShr(low, 63);
    Value* ov = B.CreateICmpNE(high, signExt, "imul.ov");
    storeFlag(LC, B, CF, ov);
    storeFlag(LC, B, OF, ov);

    recordMapping(LC, insn, low);
    return;
  }

  if (insn.operands_size() < 2) return;
  const auto& dst = insn.operands(0);
  const auto& src = insn.operands(1);
  if (!dst.has_register_()) return;
  RegInfo dri = decodeReg(dst.register_());
  if (!dri.isValid || dri.isXmm) return;

  Type* ty = intTy(LC.C, dri.bitWidth);
  Value* a = truncOrZext(B, readGprSubreg(LC, B, dri), ty);
  Value* b = resolveRValue(LC, B, src, ty, insn.id());

  Value* res = B.CreateMul(a, b, "imul.res");

  auto* smul = Intrinsic::getDeclaration(&LC.M, Intrinsic::smul_with_overflow, {ty});
  Value* call = B.CreateCall(smul, {a, b});
  Value* of = B.CreateExtractValue(call, 1);
  storeFlag(LC, B, OF, of);
  storeFlag(LC, B, CF, of);

  writeGprSubreg(LC, B, dri, res);
  recordMapping(LC, insn, res);
}

static void lowerCDQ(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Instruction& insn) {
  RegInfo eax = decodeReg("EAX");
  RegInfo edx = decodeReg("EDX");
  Value* a = readGprSubreg(LC, B, eax);
  Value* sign = B.CreateAShr(a, B.getInt32(31));
  writeGprSubreg(LC, B, edx, sign);
  recordMapping(LC, insn, sign);
}

static void lowerCDQE(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Instruction& insn) {
    RegInfo eax = decodeReg("EAX");
    RegInfo rax = decodeReg("RAX");
    Value* eaxVal = readGprSubreg(LC, B, eax);
    Value* result = B.CreateSExt(eaxVal, B.getInt64Ty(), "cdqe");
    writeGprSubreg(LC, B, rax, result);
    recordMapping(LC, insn, result);
}

static void lowerBSF_BSR(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Instruction& insn, bool isForward) {
  if (insn.operands_size() < 2) return;
  const auto& dst = insn.operands(0);
  const auto& src = insn.operands(1);

  if (!dst.has_register_()) return;
  RegInfo dri = decodeReg(dst.register_());
  if (!dri.isValid || dri.isXmm) return;

  Type* ty = intTy(LC.C, dri.bitWidth);
  Value* srcVal = resolveRValue(LC, B, src, ty, insn.id());
  if (!srcVal) return;

  // Get the appropriate count-leading/trailing zeros intrinsic
  Intrinsic::ID intrinID = isForward ? Intrinsic::cttz : Intrinsic::ctlz;
  Function* ctz = Intrinsic::getDeclaration(&LC.M, intrinID, {ty});

  // is_zero_undef = false (we want defined behavior when src == 0)
  Value* isZeroUndef = B.getFalse();
  Value* count = B.CreateCall(ctz, {srcVal, isZeroUndef}, isForward ? "bsf.cttz" : "bsr.ctlz");

  // For BSR we need: (width - 1) - ctlz(src)
  Value* result = count;
  if (!isForward) {
    Value* widthMinusOne = ConstantInt::get(ty, dri.bitWidth - 1);
    result = B.CreateSub(widthMinusOne, count, "bsr.adjust");
  }

  // ZF = (src == 0)
  Value* isZero = B.CreateICmpEQ(srcVal, ConstantInt::get(ty, 0), "bsf.zf");
  storeFlag(LC, B, ZF, isZero);

  // Other arithmetic flags are undefined
  Type* i1Ty = Type::getInt1Ty(LC.C);
  Value* undefFlag = UndefValue::get(i1Ty);
  storeFlag(LC, B, CF, undefFlag);
  storeFlag(LC, B, OF, undefFlag);
  storeFlag(LC, B, SF, undefFlag);
  storeFlag(LC, B, AF, undefFlag);
  storeFlag(LC, B, PF, undefFlag);

  // Destination: leave undefined when src == 0 (per x86 spec)
  // Use select to keep old value or write new result
  Value* oldDst = readGprSubreg(LC, B, dri);
  Value* oldDstExt = truncOrZext(B, oldDst, ty);
  Value* finalVal = B.CreateSelect(isZero, oldDstExt, result, "bsf.dst");

  writeGprSubreg(LC, B, dri, finalVal);
  recordMapping(LC, insn, finalVal);
}

static void lowerDIV(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 1) return;
  const auto& srcOp = insn.operands(0);

  RegInfo eaxRi = decodeReg("EAX");
  RegInfo edxRi = decodeReg("EDX");

  Value* eaxV = readGprSubreg(LC, B, eaxRi);
  Value* edxV = readGprSubreg(LC, B, edxRi);

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

static void lowerIDIV_ECX(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Instruction& insn) {
  if (insn.operands_size() < 1) return;
  const auto& src = insn.operands(0);

  RegInfo eax = decodeReg("EAX");
  RegInfo edx = decodeReg("EDX");
  Value* eaxV = readGprSubreg(LC, B, eax);
  Value* edxV = readGprSubreg(LC, B, edx);
  Value* hi = B.CreateSExt(edxV, B.getInt64Ty(), "idiv.hi.sext");
  Value* lo = B.CreateZExt(eaxV, B.getInt64Ty(), "idiv.lo.zext");
  Value* dividend = B.CreateOr(B.CreateShl(hi, B.getInt64(32)), lo, "idiv.dividend");

  Value* divisor32 = resolveRValue(LC, B, src, B.getInt32Ty(), insn.id());
  Value* divisor = B.CreateSExt(divisor32, B.getInt64Ty(), "idiv.divisor");

  Value* q = B.CreateSDiv(dividend, divisor, "idiv.q");
  Value* r = B.CreateSRem(dividend, divisor, "idiv.r");

  writeGprSubreg(LC, B, eax, B.CreateTrunc(q, B.getInt32Ty()));
  writeGprSubreg(LC, B, edx, B.CreateTrunc(r, B.getInt32Ty()));
  recordMapping(LC, insn, B.CreateTrunc(q, B.getInt32Ty()));
}

static void lowerUnsupportedButKeepIR(FnLowerCtx& LC, IRBuilder<>& B, const lifted_ast::Instruction& insn) {
  std::string opc = insn.has_opcode() ? toUpper(insn.opcode()) : "UNKNOWN";
  FunctionType* ft = FunctionType::get(Type::getVoidTy(LC.C), false);
  InlineAsm* ia = InlineAsm::get(ft, "; placeholder for " + opc, "", true);
  CallInst* placeholder = B.CreateCall(ia);
  recordMapping(LC, insn, placeholder);

  if (insn.operands_size() >= 1) {
    const auto& dst = insn.operands(0);
    if (dst.has_register_()) {
      RegInfo ri = decodeReg(dst.register_());
      if (ri.isValid) {
        if (ri.isXmm) {
          Type* xmmTy = FixedVectorType::get(B.getDoubleTy(), 2);
          Value* p = getStateFieldPtr(LC, B, kXmmBase + ri.xmmIndex);
          B.CreateStore(UndefValue::get(xmmTy), p);
        } else {
          Type* ty = intTy(LC.C, ri.bitWidth);
          writeGprSubreg(LC, B, ri, UndefValue::get(ty));
        }
      }
    }
  }
}

// ----------------------------- stack promotion ------------------------------

static Type* slotTypeHeuristic(LLVMContext& C, int32_t size) {
  if (size == 1) return Type::getInt8Ty(C);
  if (size == 2) return Type::getInt16Ty(C);
  if (size == 4) return Type::getInt32Ty(C);
  if (size == 8) return Type::getInt64Ty(C);
  if (size > 0) return ArrayType::get(Type::getInt8Ty(C), (uint64_t)size);
  return ArrayType::get(Type::getInt8Ty(C), 1);
}

static void promoteStackSlots(FnLowerCtx& LC, IRBuilder<>& EntryB) {
  for (const auto& ss : LC.FnAst->stack_slots()) {
    StackSlotInfo info;
    info.name = ss.has_name() ? ss.name() : "slot";
    info.baseReg = ss.has_register_() ? ss.register_() : "RBP";
    info.startOff = ss.has_offset() ? ss.offset() : 0;
    info.size = ss.has_size() ? ss.size() : 8;
    info.align = ss.has_alignment() ? ss.alignment() : 1;

    Type* ty = slotTypeHeuristic(LC.C, info.size);
    AllocaInst* a = EntryB.CreateAlloca(ty, nullptr, info.name);
    a->setAlignment(Align((unsigned)std::max<int32_t>(1, info.align)));
    info.alloca = a;
    LC.promotedSlots.push_back(info);
  }
}

// ----------------------------- per-function lowering ------------------------

static void lowerFunction(FnLowerCtx& LC) {
  auto origLinkage = LC.F->getLinkage();
  LC.F->deleteBody();

  for (const auto& bbAst : LC.FnAst->basic_blocks()) {
    std::string bbName = bbAst.has_start_label() && !bbAst.start_label().empty() ? bbAst.start_label() : (bbAst.has_id() && !bbAst.id().empty() ? bbAst.id() : "bb");
    BasicBlock* bb = BasicBlock::Create(LC.C, bbName, LC.F);
    if (bbAst.has_id()) {
      LC.bbIdToLlvm[bbAst.id()] = bb;
      if (LC.bbLlvmMapping) (*LC.bbLlvmMapping)[bbAst.id()] = bb->getName().str();
    }
  }

  LC.StateArg = LC.F->getArg(0);
  if (LC.FnAst->basic_blocks_size() > 0) {
    const auto& entryAst = LC.FnAst->basic_blocks(0);
    if (auto it = LC.bbIdToLlvm.find(entryAst.id()); it != LC.bbIdToLlvm.end()) {
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
      std::string opc = insn.has_opcode() ? toUpper(insn.opcode()) : "";

      if (bbAst.has_terminator() && insn.has_id() && insn.id() == bbAst.terminator()) continue;
      if (isTerminatorOpcode(opc) || isFloatOpcode(opc)) continue;

      if (opc == "MOV") lowerMOV(LC, B, insn);
      else if (opc == "LEA") lowerLEA(LC, B, insn);
      else if (opc == "ADD" || opc == "SUB" || opc == "XOR" || opc == "AND" || opc == "OR") lowerALU2(LC, B, insn, opc);
      else if (opc == "CMP") lowerCMP(LC, B, insn);
      else if (opc == "TEST") lowerTEST(LC, B, insn);
      else if (opc == "MOVZX") lowerMOVZX_MOVSX(LC, B, insn, false);
      else if (opc == "MOVSX") lowerMOVZX_MOVSX(LC, B, insn, true);
      else if (opc == "SETC" || opc == "SETB" || opc == "SETNE" || opc == "SETNZ" || opc == "SETPE" || opc == "SETP" || opc == "SETE" || opc == "SETZ") lowerSETcc(LC, B, insn, opc);
      else if (opc == "CMOVE") lowerCMOVE(LC, B, insn);
      else if (opc == "INC") lowerINC(LC, B, insn);
      else if (opc == "SHL") lowerSHL(LC, B, insn);
      else if (opc == "SHR") lowerSHR(LC, B, insn);
      else if (opc == "SAR") lowerSAR(LC, B, insn);
      else if (opc == "XCHG") lowerXCHG(LC, B, insn);
      else if (opc == "XADD") lowerXADD(LC, B, insn);
      else if (opc == "PUSH") lowerPUSH(LC, B, insn);
      else if (opc == "POP") lowerPOP(LC, B, insn);
      else if (opc == "LEAVE") lowerLEAVE(LC, B, insn);
      else if (opc == "IMUL") lowerIMUL(LC, B, insn);
      else if (opc == "CDQ") lowerCDQ(LC, B, insn);
      else if (opc == "CDQE") lowerCDQE(LC, B, insn);
      else if (opc == "BSF") lowerBSF_BSR(LC, B, insn, true);
      else if (opc == "BSR") lowerBSF_BSR(LC, B, insn, false);
      else if (opc == "DIV")  lowerDIV(LC, B, insn);
      else if (opc == "IDIV") lowerIDIV_ECX(LC, B, insn);
      else lowerUnsupportedButKeepIR(LC, B, insn);
    }
    if (!bb->getTerminator()) B.CreateUnreachable();
  }

  LC.F->setLinkage(origLinkage);

  if (verifyFunction(*LC.F, &errs())) errs() << "verifyFunction failed for: " << LC.F->getName() << "\n";
}

// ----------------------------- module driver -------------------------------

static std::unique_ptr<Module> loadBitcodeModule(const std::string& path, LLVMContext& C) {
  auto bufOrErr = MemoryBuffer::getFile(path);
  if (!bufOrErr) { std::cerr << "Failed to open bitcode: " << path << "\n"; return nullptr; }
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

  LLVMContext C;
  std::unique_ptr<Module> M = loadBitcodeModule(args[0], C);
  if (!M) return 1;

  StructType* StateTy = StructType::getTypeByName(C, "State");
  if (!StateTy) { std::cerr << "ERROR: Could not find identified struct type 'State' in module.\n"; return 1; }

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
      if (itSe != P.symbol_table().end() && itSe->second.has_lifted_ref() && !itSe->second.lifted_ref().empty()) {
        liftedName = itSe->second.lifted_ref();
      }

      if (Function* F = M->getFunction(liftedName)) {
        FnLowerCtx LC{C, *M, StateTy, F, nullptr, &P, &fnAst, instrLlvmMapping, &bbLlvmMapping};
        lowerFunction(LC);
      } else {
        errs() << "Warning: lifted function not found in module: " << liftedName << "\n";
      }
    }
  }

  for (const auto& kv : bbLlvmMapping) (*P.mutable_bb_llvm_mapping())[kv.first] = kv.second;
  for (const auto& kv : instrLlvmMapping) (*P.mutable_instr_llvm_mapping())[kv.first] = kv.second;

  std::string originalDL = M->getDataLayoutStr();
  std::string originalTriple = M->getTargetTriple();
  if (verifyModule(*M, &errs())) errs() << "verifyModule FAILED after Step 11 lowering.\n";
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
    raw_fd_ostream os(args[2], EC, sys::fs::OF_None);
    if (EC) return 1;
    if (printMode) M->print(os, nullptr);
    else WriteBitcodeToFile(*M, os);
    os.flush();
  }

  if (!printIrPath.empty()) {
    std::error_code EC;
    raw_fd_ostream os(printIrPath, EC, sys::fs::OF_None);
    if (!EC) { M->print(os, nullptr); os.flush(); }
  }
  return 0;
}
