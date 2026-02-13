"""
llvmInit.py
Step 10: LLVM Module Skeleton Generation.
"""
import sys
import json
import re
import ast  # For parsing array literals
import struct  # For robust float/double bit patterns
from llvmlite import ir
from astNodes import legacy_program_dict_to_ast, Program, Function

# Modern Opaque Pointer
ptr = ir.PointerType()

def get_ir_type_from_string(type_str: str):
    """Map common lifted return type strings to llvmlite types."""
    type_str = type_str.strip()
    if type_str == 'void':
        return ir.VoidType()
    if type_str == 'i64':
        return ir.IntType(64)
    if type_str == 'i32':
        return ir.IntType(32)
    if type_str == 'float':
        return ir.FloatType()
    if type_str == 'double':
        return ir.DoubleType()
    return ir.IntType(64)

def create_external_function(module: ir.Module, name: str):
    """Signatures for known externals using opaque ptr."""
    i32 = ir.IntType(32)
    if name == 'printf':
        ftype = ir.FunctionType(i32, [ptr], var_arg=True)
    elif name == 'fprintf':
        ftype = ir.FunctionType(i32, [ptr, ptr], var_arg=True)
    elif name == 'exit':
        ftype = ir.FunctionType(ir.VoidType(), [i32])
    elif name == 'abort':
        ftype = ir.FunctionType(ir.VoidType(), [])
    else:
        # Unknown external: conservative i64 return
        ftype = ir.FunctionType(ir.IntType(64), [ptr], var_arg=True)
    func = ir.Function(module, ftype, name=name)
    func.linkage = 'external'
    if name in ('exit', 'abort'):
        func.attributes.add('noreturn')
    return func

def make_float_constant(value: str):
    """Robust single-precision constant from string value."""
    f = float(value)
    packed = struct.pack('!f', f)          # network byte order
    bits = struct.unpack('!I', packed)[0]
    int_c = ir.Constant(ir.IntType(32), bits)
    return int_c.bitcast(ir.FloatType())

def make_double_constant(value: str):
    """Robust double-precision constant from string value."""
    d = float(value)
    packed = struct.pack('!d', d)
    bits = struct.unpack('!Q', packed)[0]
    int_c = ir.Constant(ir.IntType(64), bits)
    return int_c.bitcast(ir.DoubleType())

def _fix_typed_func_ptrs(ir_text):
    """Replace typed function pointers in llvm.global_ctors/dtors with opaque ptr.

    llvmlite may emit  ``void ()* @func``  instead of  ``ptr @func``  inside
    aggregate constant initializers.  This post-processing pass rewrites those
    occurrences on the relevant lines to satisfy the opaque-pointer verifier.
    """
    lines = ir_text.splitlines()
    for i, line in enumerate(lines):
        if 'llvm.global_ctors' in line or 'llvm.global_dtors' in line:
            # Pattern: "rettype (params)* @name" → "ptr @name"
            lines[i] = re.sub(r'\w+\s*\([^)]*\)\*(?=\s+@)', 'ptr', line)
    return '\n'.join(lines)

def main():
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            sys.stderr.write("No input on stdin.\n")
            sys.exit(1)
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"JSON parse error: {e}\n")
        sys.exit(1)

    program = legacy_program_dict_to_ast(data, include_enhancements=True)
    symbol_table = program.symbol_table

    # Create LLVM module
    module = ir.Module()
    module.triple = 'x86_64-unknown-linux-gnu'
    module.data_layout = 'e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128'

    # Define %State struct
    state_type = module.context.get_identified_type('State')
    i64 = ir.IntType(64)
    i1 = ir.IntType(1)
    vec2double = ir.VectorType(ir.DoubleType(), 2)
    state_type.set_body(
        *[i64] * 16,
        *[i1] * 9,
        *[vec2double] * 16
    )

    # First pass: internal lifted functions – uniform %State* threading
    # NOTE: Constructor/destructor stubs now follow the same convention as all
    # other lifted functions (void(ptr)) to maintain internal ABI consistency.
    for sym, info in symbol_table.items():
        if not (info.get('is_definition') and info.get('kind') in ('label', 'function')):
            continue

        func_node = None
        for sec in program.sections:
            if sec.name != '.text':
                continue
            for child in sec.children:
                if isinstance(child, Function) and (child.entry_label == sym or child.id == f"func:{sym}"):
                    func_node = child
                    break
            if func_node:
                break

        if not func_node or not getattr(func_node, 'lifted_signature', None):
            continue

        ret_str = func_node.lifted_signature.return_type
        ret_type = get_ir_type_from_string(ret_str)

        # Uniform state-threaded signature for ALL lifted functions (including ctors/dtors)
        ftype = ir.FunctionType(ret_type, [ptr])

        func = ir.Function(module, ftype, name=sym)
        func.linkage = 'internal'

        if func_node.lifted_signature.attributes:
            for attr in func_node.lifted_signature.attributes:
                if attr == 'noreturn':
                    func.attributes.add('noreturn')

        info['llvm_ref'] = func

    # Second pass: external functions
    for sym, info in symbol_table.items():
        if info.get('is_external') and info.get('kind') == 'function':
            func = create_external_function(module, sym)
            info['llvm_ref'] = func

    # Third pass: globals (data + constants)
    for sym, info in symbol_table.items():
        if info.get('kind') not in ('data', 'constant'):
            continue

        linkage = info.get('linkage', 'internal')
        is_constant = info.get('is_constant', False)
        section = info.get('section', '')
        value = info.get('value', '')
        llvm_type_str = info.get('llvm_type', '')

        # External data (e.g., stderr)
        if info.get('is_external'):
            gv = ir.GlobalVariable(module, ptr, name=sym)
            gv.linkage = 'external'
            info['llvm_ref'] = gv
            continue

        # Assembler equ constants → compile-time immediates, no global emitted
        if info.get('kind') == 'constant':
            info['llvm_immediate'] = int(value)
            continue

        # Regular data globals
        if llvm_type_str.startswith('['):
            match = re.match(r'\[(\d+) x i(\d+)\]', llvm_type_str)
            if match:
                count, width = int(match.group(1)), int(match.group(2))
                elem_type = ir.IntType(width)
                initializer = None
                actual_size = count

                if value.startswith('c"'):
                    s = value[2:-1]
                    s = s.replace(r'\0A', r'\n')
                    s = bytes(s, "utf-8").decode("unicode_escape")
                    bytes_val = s.encode('utf-8')
                    init_bytes = bytearray(bytes_val)
                    if not s.endswith('\0'):
                        init_bytes.append(0)
                    actual_size = len(init_bytes)
                    initializer = ir.Constant(ir.ArrayType(elem_type, actual_size), init_bytes)
                elif value.startswith('['):
                    try:
                        arr_list = ast.literal_eval(value)
                        if isinstance(arr_list, list):
                            int_list = [int(x) for x in arr_list]
                            actual_size = len(int_list)
                            array_type = ir.ArrayType(elem_type, actual_size)
                            initializer = ir.Constant(array_type, int_list)
                    except Exception:
                        pass

                array_type = ir.ArrayType(elem_type, actual_size)
                gv = ir.GlobalVariable(module, array_type, name=sym)
                gv.linkage = linkage
                if is_constant:
                    gv.global_constant = True
                if initializer:
                    gv.initializer = initializer
                elif section == '.bss' or value == 'zeroinitializer':
                    gv.initializer = ir.Constant(gv.type.pointee, None)
                info['llvm_ref'] = gv
                continue

        elif llvm_type_str == 'float':
            gv = ir.GlobalVariable(module, ir.FloatType(), name=sym)
            gv.linkage = linkage
            if is_constant:
                gv.global_constant = True
            if section == '.bss' or value == 'zeroinitializer' or not value:
                gv.initializer = ir.Constant(ir.FloatType(), None)
            else:
                gv.initializer = make_float_constant(value)
            info['llvm_ref'] = gv
            continue

        elif llvm_type_str == 'double':
            gv = ir.GlobalVariable(module, ir.DoubleType(), name=sym)
            gv.linkage = linkage
            if is_constant:
                gv.global_constant = True
            if section == '.bss' or value == 'zeroinitializer' or not value:
                gv.initializer = ir.Constant(ir.DoubleType(), None)
            else:
                gv.initializer = make_double_constant(value)
            info['llvm_ref'] = gv
            continue

        elif 'i32' in llvm_type_str or 'i64' in llvm_type_str:
            bitwidth = 32 if 'i32' in llvm_type_str else 64
            gv = ir.GlobalVariable(module, ir.IntType(bitwidth), name=sym)
            gv.linkage = linkage
            if section == '.bss' or value == 'zeroinitializer':
                gv.initializer = ir.Constant(gv.type.pointee, None)
            else:
                gv.initializer = ir.Constant(gv.type.pointee, int(value))
            info['llvm_ref'] = gv
            continue

        else:
            gv = ir.GlobalVariable(module, ptr, name=sym)
            gv.linkage = linkage
            info['llvm_ref'] = gv

    # llvm.global_ctors / dtors
    # Note: The function pointers here are opaque 'ptr' type, which is compatible
    # with the lifted function signatures (the runtime will need to pass the state
    # pointer when invoking these).
    if 'llvm.global_ctors' in symbol_table:
        elem = ir.LiteralStructType([ir.IntType(32), ptr, ptr])
        array_type = ir.ArrayType(elem, 1)
        gv = ir.GlobalVariable(module, array_type, 'llvm.global_ctors')
        gv.linkage = 'appending'
        priority = ir.Constant(ir.IntType(32), 65535)
        func = module.get_global('constructor_stub')
        func_ptr = func.bitcast(ptr)
        null = ir.Constant(ptr, None)
        init_elem = ir.Constant(elem, (priority, func_ptr, null))
        gv.initializer = ir.Constant(array_type, [init_elem])

    if 'llvm.global_dtors' in symbol_table:
        elem = ir.LiteralStructType([ir.IntType(32), ptr, ptr])
        array_type = ir.ArrayType(elem, 1)
        gv = ir.GlobalVariable(module, array_type, 'llvm.global_dtors')
        gv.linkage = 'appending'
        priority = ir.Constant(ir.IntType(32), 65535)
        func = module.get_global('destructor_stub')
        func_ptr = func.bitcast(ptr)
        null = ir.Constant(ptr, None)
        init_elem = ir.Constant(elem, (priority, func_ptr, null))
        gv.initializer = ir.Constant(array_type, [init_elem])

    # PIC module flag
    int32 = ir.IntType(32)
    pic_level = module.add_metadata([
        ir.Constant(int32, 1),
        ir.MetaDataString(module.context, "PIC Level"),
        ir.Constant(int32, 2)
    ])
    module.add_named_metadata('llvm.module.flags', pic_level)

    # Emit IR, fixing any residual typed function pointers from llvmlite
    output = _fix_typed_func_ptrs(str(module))
    print(output)

if __name__ == '__main__':
    main()
