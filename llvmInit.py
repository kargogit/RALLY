import sys
import json
from astNodes import legacy_program_dict_to_ast
from llvmUtils import (
    build_llvm_skeleton, LiftingContext,
    fix_float_literals, sanitize_ir, append_pic_metadata
)

def main():
    raw = sys.stdin.read().strip()
    if not raw:
        print("No input on stdin", file=sys.stderr)
        sys.exit(1)
    obj = json.loads(raw)
    program = legacy_program_dict_to_ast(obj, include_enhancements=True)

    module, ctx = build_llvm_skeleton(program)
    ctx.serialized_ast = obj   # so Step 11+ can rebuild instantly

    ir_str = str(module)
    ir_str = sanitize_ir(ir_str)
    ir_str = fix_float_literals(ir_str, program.symbol_table)
    ir_str = append_pic_metadata(ir_str)

    ctx.module_ir_str = ir_str
    LiftingContext.save(ctx)

    print(ir_str)

if __name__ == "__main__":
    main()
