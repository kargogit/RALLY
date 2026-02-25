import json
import os
import re
import struct
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from llvmlite import ir

from astNodes import Program, Function, legacy_program_dict_to_ast

# Add these two helpers near the top (after imports, before StateLayout)
def _normalize_llvm_type(llvm_type: str) -> str:
    """Convert legacy typed pointers to opaque 'ptr' (LLVM 15+ style)."""
    if not isinstance(llvm_type, str):
        return llvm_type
    s = llvm_type
    s = re.sub(r'\bi\d+\*', 'ptr', s)
    s = s.replace('i8*', 'ptr').replace('float*', 'ptr').replace('double*', 'ptr')
    s = re.sub(r'void \(\)\*', 'ptr', s)
    # Function signatures: i32 (i8*, i8*, ...)  →  i32 (ptr, ptr, ...)
    s = re.sub(r'([,(]\s*)i\d+\*', r'\1ptr', s)
    s = re.sub(r'([,(]\s*)i8\*', r'\1ptr', s)
    s = re.sub(r'([,(]\s*)float\*', r'\1ptr', s)
    s = re.sub(r'([,(]\s*)double\*', r'\1ptr', s)
    return s

def _normalize_ctor_dtor_value(value: Any) -> Any:
    """Convert legacy typed pointers inside llvm.global_ctors / llvm.global_dtors value strings."""
    if not isinstance(value, str):
        return value
    v = value
    v = v.replace("void ()*", "ptr")
    v = v.replace("i8* null", "ptr null")   # more precise than blind i8* replace
    return v


def normalize_symbol_table_types(symbol_table: Dict[str, Any]) -> None:
    """In-place normalization — now covers llvm_type *and* ctor/dtor value strings."""
    for name, info in symbol_table.items():
        if not isinstance(info, dict):
            continue

        # llvm_type normalization
        if "llvm_type" in info:
            info["llvm_type"] = _normalize_llvm_type(info["llvm_type"])

        # special-case the synthetic global_ctors / global_dtors value strings
        if name in ("llvm.global_ctors", "llvm.global_dtors") and "value" in info:
            info["value"] = _normalize_ctor_dtor_value(info["value"])


class StateLayout:
    GPR_RAX = 0
    GPR_RCX = 1
    GPR_RDX = 2
    GPR_RBX = 3
    GPR_RSP = 4
    GPR_RBP = 5
    GPR_RSI = 6
    GPR_RDI = 7
    GPR_R8 = 8
    GPR_R9 = 9
    GPR_R10 = 10
    GPR_R11 = 11
    GPR_R12 = 12
    GPR_R13 = 13
    GPR_R14 = 14
    GPR_R15 = 15
    FLAG_CF = 16
    FLAG_PF = 17
    FLAG_AF = 18
    FLAG_ZF = 19
    FLAG_SF = 20
    FLAG_TF = 21
    FLAG_IF = 22
    FLAG_DF = 23
    FLAG_OF = 24
    XMM0 = 25
    XMM1 = 26
    XMM2 = 27
    XMM3 = 28
    XMM4 = 29
    XMM5 = 30
    XMM6 = 31
    XMM7 = 32
    XMM8 = 33
    XMM9 = 34
    XMM10 = 35
    XMM11 = 36
    XMM12 = 37
    XMM13 = 38
    XMM14 = 39
    XMM15 = 40

    @classmethod
    def gpr_name_to_index(cls, name: str) -> Optional[int]:
        mapping = {k.lower(): v for k, v in {
            'rax': 0, 'rcx': 1, 'rdx': 2, 'rbx': 3, 'rsp': 4, 'rbp': 5,
            'rsi': 6, 'rdi': 7, 'r8': 8, 'r9': 9, 'r10': 10, 'r11': 11,
            'r12': 12, 'r13': 13, 'r14': 14, 'r15': 15,
            'eax': 0, 'ecx': 1, 'edx': 2, 'ebx': 3, 'esp': 4, 'ebp': 5,
            'esi': 6, 'edi': 7, 'r8d': 8, 'r9d': 9, 'r10d': 10, 'r11d': 11,
            'r12d': 12, 'r13d': 13, 'r14d': 14, 'r15d': 15,
        }.items()}
        return mapping.get(name.lower())


@dataclass
class LiftingContext:
    module_ir_str: Optional[str] = None
    serialized_ast: Optional[Dict[str, Any]] = None   # full Step-9 AST (for Step 11+)
    ast_to_llvm_map: Dict[str, str] = field(default_factory=dict)
    #symbol_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'LiftingContext':
        return LiftingContext(**d)

    @staticmethod
    def save(ctx: 'LiftingContext', path: str = "lifting_context_step10.json"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(ctx.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str = "lifting_context_step10.json") -> 'LiftingContext':
        with open(path) as f:
            return LiftingContext.from_dict(json.load(f))


# ===================================================================
# Skeleton builder (used by Step 10 and reusable by Step 11+)
# ===================================================================
def define_state_struct(module: ir.Module) -> ir.IdentifiedStructType:
    state_ty = module.context.get_identified_type("State")
    gpr = ir.IntType(64)
    flag = ir.IntType(1)
    xmm = ir.VectorType(ir.DoubleType(), 2)
    body = [gpr] * 16 + [flag] * 9 + [xmm] * 16
    state_ty.set_body(*body)
    return state_ty


def parse_llvm_type_str(s: str, module: ir.Module) -> ir.Type:
    s = s.strip()
    if s == "void": return ir.VoidType()
    if s in ("ptr", "i8*", "i8 *"): return ir.PointerType()
    if s.startswith("i") and s[1:].isdigit(): return ir.IntType(int(s[1:]))
    if s == "float": return ir.FloatType()
    if s in ("double", "f64"): return ir.DoubleType()

    # Function type
    if "(" in s and s.endswith(")"):
        ret_part, args_part = s.split(" (", 1)
        ret_ty = parse_llvm_type_str(ret_part.strip(), module)
        args_str = args_part.rstrip(")").strip()
        arg_tys = []
        vararg = False
        if args_str and args_str != "void":
            for p in [x.strip() for x in args_str.split(",")]:
                if p == "...":
                    vararg = True
                    continue
                arg_tys.append(parse_llvm_type_str(p, module))
        return ir.FunctionType(ret_ty, arg_tys, vararg)

    # Array [N x T]
    if s.startswith("[") and s.endswith("]") and " x " in s:
        inner = s[1:-1].strip()
        count_str, elem_str = inner.split(" x ", 1)
        count = int(count_str.strip())
        elem_ty = parse_llvm_type_str(elem_str.strip(), module)
        return ir.ArrayType(elem_ty, count)

    raise ValueError(f"Unsupported LLVM type string: '{s}'")


def _unescape_llvm_cstring(escaped: str) -> bytes:
    b = bytearray()
    i = 0
    n = len(escaped)
    while i < n:
        if escaped[i] == '\\' and i + 1 < n:
            i += 1
            ch = escaped[i]
            if ch == '0' and i + 1 < n and escaped[i+1].lower() in '0123456789abcdef':
                b.append(int('0' + escaped[i+1], 16))
                i += 2
                continue
            if ch == 'n': b.append(10)
            elif ch == '\\': b.append(ord('\\'))
            elif ch == '"': b.append(ord('"'))
            elif ch.isdigit():
                oct_str = ch
                i += 1
                while i < n and escaped[i].isdigit() and len(oct_str) < 3:
                    oct_str += escaped[i]
                    i += 1
                b.append(int(oct_str, 8))
                continue
            else:
                b.append(ord(ch))
            i += 1
        else:
            b.append(ord(escaped[i]))
            i += 1
    return bytes(b)


def get_initializer(typ: ir.Type, value: Any, section: str) -> ir.Constant:
    if value == "zeroinitializer" or section == ".bss":
        return ir.Constant(typ, None)

    try:
        if isinstance(typ, ir.ArrayType) and typ.element == ir.IntType(8):
            if isinstance(value, str):
                v = value.strip()
                if v.startswith('c"') and v.endswith('"'):
                    inner = v[2:-1]
                    b = _unescape_llvm_cstring(inner)
                    if not b or b[-1] != 0:
                        b += b'\0'
                    if len(b) < typ.count:
                        b += b'\0' * (typ.count - len(b))
                    return ir.Constant(typ, bytearray(b[:typ.count]))
                elif v.startswith('[') and v.endswith(']'):
                    inner = v[1:-1].strip()
                    nums = [int(x.strip()) for x in inner.split(",") if x.strip()]
                    consts = [ir.Constant(typ.element, n) for n in nums]
                    if len(consts) < typ.count:
                        consts += [ir.Constant(typ.element, 0)] * (typ.count - len(consts))
                    return ir.Constant(typ, consts)

        if isinstance(typ, ir.IntType):
            return ir.Constant(typ, int(value))
        if isinstance(typ, (ir.FloatType, ir.DoubleType)):
            return ir.Constant(typ, float(value))
    except Exception:
        pass
    return ir.Constant(typ, None)


def create_globals_and_externals(module: ir.Module, symbol_table: Dict[str, Any]):
    for name, info in symbol_table.items():
        if name.startswith("llvm.global_"):
            continue
        kind = info.get("kind")
        linkage = info.get("linkage", "internal")
        llvm_type_str = info.get("llvm_type", "i8*")
        is_external = info.get("is_external", False)
        section = info.get("section", "")
        typ = parse_llvm_type_str(llvm_type_str, module)
        if kind == "data":
            gv = ir.GlobalVariable(module, typ, name=name)
            gv.linkage = "external" if is_external else linkage
            if not is_external:
                val = info.get("value")
                gv.initializer = get_initializer(typ, val, section)
                gv.dso_local = True
        elif kind == "function" and is_external:
            f = ir.Function(module, typ, name=name)
            f.linkage = "external"
            f.attributes.add("nounwind")
            if name == "exit":
                f.attributes.add("noreturn")


def add_global_ctors_dtors(module: ir.Module, symbol_table: Dict[str, Any]):
    i32 = ir.IntType(32)
    ptr = ir.PointerType()
    struct_ty = ir.LiteralStructType([i32, ptr, ptr])
    for key in ("llvm.global_ctors", "llvm.global_dtors"):
        if key not in symbol_table:
            continue
        wrapper_name = "constructor_stub" if "ctors" in key else "destructor_stub"
        wrapper = module.get_global(wrapper_name) if hasattr(module, "get_global") else None
        if wrapper is None:
            continue
        arr_ty = ir.ArrayType(struct_ty, 1)
        gv = ir.GlobalVariable(module, arr_ty, name=key)
        gv.linkage = "appending"
        gv.dso_local = True
        prio = ir.Constant(i32, 65535)
        null_ptr = ir.Constant(ptr, None)
        entry = ir.Constant(struct_ty, (prio, wrapper, null_ptr))
        gv.initializer = ir.Constant(arr_ty, [entry])


def synthesize_wrapper(module: ir.Module, f: Function, state_ty: ir.Type, lifted_fn: ir.Function, wrapper_linkage: str = "external"):
    if not getattr(f, "is_boundary", False) or not f.external_abi_signature:
        return
    wrapper_ty = parse_llvm_type_str(f.external_abi_signature, module)
    wrapper = ir.Function(module, wrapper_ty, name=f.entry_label)
    wrapper.linkage = wrapper_linkage
    wrapper.attributes.add("nounwind")
    if f.lifted_signature:
        for attr in f.lifted_signature.attributes:
            wrapper.attributes.add(attr)
    entry = wrapper.append_basic_block("entry")
    builder = ir.IRBuilder(entry)
    state_ptr = builder.alloca(state_ty, name="state")
    for i, arg_val in enumerate(wrapper.args):
        if i < 6:
            gpr_idx = [StateLayout.GPR_RDI, StateLayout.GPR_RSI, StateLayout.GPR_RDX,
                       StateLayout.GPR_RCX, StateLayout.GPR_R8, StateLayout.GPR_R9][i]
            gpr_ptr = builder.gep(state_ptr, [ir.Constant(ir.IntType(32), 0),
                                              ir.Constant(ir.IntType(32), gpr_idx)])
            if isinstance(arg_val.type, ir.IntType) and arg_val.type.width < 64:
                builder.store(builder.zext(arg_val, ir.IntType(64)), gpr_ptr)
            elif isinstance(arg_val.type, ir.PointerType):
                builder.store(builder.ptrtoint(arg_val, ir.IntType(64)), gpr_ptr)
            else:
                builder.store(arg_val, gpr_ptr)
    builder.call(lifted_fn, [state_ptr])
    is_noreturn = f.lifted_signature and "noreturn" in f.lifted_signature.attributes
    ret_ty = wrapper_ty.return_type
    if is_noreturn:
        builder.unreachable()
    elif isinstance(ret_ty, ir.VoidType):
        builder.ret_void()
    else:
        if isinstance(ret_ty, ir.IntType):
            rax_ptr = builder.gep(state_ptr, [ir.Constant(ir.IntType(32), 0),
                                              ir.Constant(ir.IntType(32), StateLayout.GPR_RAX)])
            val = builder.load(rax_ptr)
            if ret_ty.width < 64:
                val = builder.trunc(val, ret_ty)
            builder.ret(val)
        else:
            builder.ret(ir.Constant(ret_ty, None))


def fix_float_literals(ir_str: str, symbol_table: Dict[str, Any] = None) -> str:
    def replacer(m: re.Match) -> str:
        prefix = m.group(1)
        hex_val = m.group(2)
        if symbol_table:
            before = m.string[:m.start()].rsplit('\n', 1)[-1]
            name_m = re.search(r'@\"([^\"]+)\"', before)
            if name_m:
                name = name_m.group(1)
                info = symbol_table.get(name, {})
                if info.get("llvm_type") == "float":
                    val = info.get("value")
                    if val is not None:
                        try:
                            return f"{prefix}{float(val):g}"
                        except Exception:
                            pass
        try:
            bits = int(hex_val[2:], 16)
            d = struct.unpack('>d', struct.pack('>Q', bits))[0]
            f = float(d)
            return f"{prefix}{f:g}" if f != int(f) else f"{prefix}{int(f)}.0"
        except Exception:
            return m.group(0)
    return re.sub(r'(float\s+)(0x[0-9a-fA-F]{16})', replacer, ir_str)


def sanitize_ir(ir_str: str) -> str:
    ir_str = ir_str.replace('%"State"*', 'ptr')
    ir_str = re.sub(r'\bi\d+\*', 'ptr', ir_str)
    ir_str = ir_str.replace('float*', 'ptr').replace('double*', 'ptr')
    ir_str = ir_str.replace('void ()*', 'ptr')
    ir_str = re.sub(r'\[\d+\s+x\s+[^\]]+\]\*', 'ptr', ir_str)
    ir_str = re.sub(r'\{[^\}]+\}\*', 'ptr', ir_str)
    return ir_str


def append_pic_metadata(ir_str: str) -> str:
    if "!llvm.module.flags" in ir_str:
        return ir_str
    max_id = -1
    for match in re.finditer(r'^!(\d+)\s*=', ir_str, re.MULTILINE):
        max_id = max(max_id, int(match.group(1)))
    id1 = max_id + 1
    id2 = max_id + 2
    md = f"\n!llvm.module.flags = !{{!{id1}, !{id2}}}\n!{id1} = !{{i32 7, !\"PIC Level\", i32 2}}\n!{id2} = !{{i32 7, !\"PIE Level\", i32 2}}\n"
    return ir_str + md


def build_llvm_skeleton(program: Program) -> Tuple[ir.Module, LiftingContext]:
    normalize_symbol_table_types(program.symbol_table)
    module = ir.Module(name="lifted")
    module.triple = "x86_64-unknown-linux-gnu"
    module.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

    state_ty = define_state_struct(module)
    state_ptr_ty = ir.PointerType()

    create_globals_and_externals(module, program.symbol_table)

    ctx = LiftingContext(config={"target_triple": module.triple})

    for sec in program.sections:
        for child in sec.children:
            if not isinstance(child, Function):
                continue
            f = child
            name = f.entry_label
            lift_ret_str = f.lifted_signature.return_type if f.lifted_signature else "i64"
            lift_ret = parse_llvm_type_str(lift_ret_str, module)
            lifted_ty = ir.FunctionType(lift_ret, [state_ptr_ty])
            lifted_name = f"{name}_lifted" if f.is_boundary else name
            lifted_fn = ir.Function(module, lifted_ty, lifted_name)
            lifted_fn.linkage = "internal"
            lifted_fn.attributes.add("nounwind")
            if f.lifted_signature:
                for attr in f.lifted_signature.attributes:
                    lifted_fn.attributes.add(attr)

            ctx.ast_to_llvm_map[f.id] = lifted_name

            if f.is_boundary:
                wrapper_linkage = program.symbol_table.get(name, {}).get("linkage", "external")
                synthesize_wrapper(module, f, state_ty, lifted_fn, wrapper_linkage)
                if name in program.symbol_table:
                    program.symbol_table[name]["llvm_ref"] = name
                    program.symbol_table[name]["lifted_ref"] = lifted_name
            else:
                if name in program.symbol_table:
                    program.symbol_table[name]["llvm_ref"] = lifted_name

    add_global_ctors_dtors(module, program.symbol_table)
    return module, ctx
