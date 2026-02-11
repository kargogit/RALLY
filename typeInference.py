"""
typeInference.py
Step 8: Intraprocedural Data-Flow Analysis for Type Refinement.

Performs lightweight, iterative fixed-point data-flow analysis over function basic blocks
to propagate and refine type information for registers (including subregisters), scalar
operations, operands, memory accesses, and stack slots.

Infers and annotates:
- Operation refinements (op_refinement) on instructions: integer width/signedness/neutral,
  pointer vs. arithmetic distinction, basic scalar floating-point (f32/f64).
- Operand types (inferred_type): detailed scalar integer types, signedness, pointers,
  and floating-point.
- Special handling for x86-64 idioms: subregister dependencies, zero-extension,
  pointer arithmetic (ADD/SUB/INC/DEC), LEA semantics, floating-point conversions,
  SSE scalar ops, sign/zero-extension moves, and opcode-specific behaviors.

Uses a least-upper-bound (LUB) lattice for type merging with conservative fallbacks.

Inputs: Enriched AST from Step 7 (via stdin JSON).
Outputs: Further enriched AST with type refinement annotations (via stdout JSON).
"""

import sys
import json
from collections import defaultdict
from typing import Dict
from astNodes import (
    legacy_program_dict_to_ast,
    ast_to_legacy_program_dict,
    Program,
    Function,
    Instruction,
    Operand,
    StackSlot,
)
SIZE_MAP = {
    "BYTE": 8,
    "WORD": 16,
    "DWORD": 32,
    "QWORD": 64,
    None: 64,
}
def get_reg_width(reg: str) -> int:
    if reg is None:
        return 64
    if reg.startswith("R") or reg in ("RIP", "RSP", "RBP"):
        return 64
    if reg.startswith("E") or reg in ("ESP", "EBP"):
        return 32
    # Moved before the 16-bit check to correctly handle AL/AH/BL etc.
    if reg.endswith("L") or reg.endswith("H") or reg.endswith("B"):
        return 8
    if len(reg) == 2 and reg.isalpha(): # AX, BX, etc.
        return 16
    return 64
def is_zeroing_write(reg: str) -> bool:
    width = get_reg_width(reg)
    return width in (16, 32)
def lub(a: str, b: str) -> str:
    if a == "unknown":
        return b
    if b == "unknown":
        return a
    if a == b:
        return a
    # ptr conflicts or mixes with non-ptr -> conservative i64
    if "ptr" in (a, b):
        if a == "ptr" and b == "ptr":
            return "ptr"
        return "i64"
    # float conflicts -> unknown
    if a.startswith("f") or b.startswith("f"):
        return "unknown"
    # integer cases
    a_width = int(a[1:].split("_")[0]) if "_" in a[1:] else int(a[1:])
    b_width = int(b[1:].split("_")[0]) if "_" in b[1:] else int(b[1:])
    width = max(a_width, b_width)
    a_sign = a.split("_")[1] if "_" in a else "neutral"
    b_sign = b.split("_")[1] if "_" in b else "neutral"
    sign = "neutral" if a_sign != b_sign else a_sign
    return f"i{width}" if sign == "neutral" else f"i{width}_{sign}"
class Step8Refinement:
    def __init__(self, program: Program):
        self.program = program
    def run(self):
        for section in self.program.sections:
            for child in section.children:
                if isinstance(child, Function):
                    self.analyze_function(child)
    def analyze_function(self, func: Function):
        sub_to_full = {
            "AL": "RAX", "AH": "RAX", "AX": "RAX", "EAX": "RAX", "RAX": "RAX",
            "BL": "RBX", "BH": "RBX", "BX": "RBX", "EBX": "RBX", "RBX": "RBX",
            "CL": "RCX", "CH": "RCX", "CX": "RCX", "ECX": "RCX", "RCX": "RCX",
            "DL": "RDX", "DH": "RDX", "DX": "RDX", "EDX": "RDX", "RDX": "RDX",
            "SIL": "RSI", "SI": "RSI", "ESI": "RSI", "RSI": "RSI",
            "DIL": "RDI", "DI": "RDI", "EDI": "RDI", "RDI": "RDI",
            "BPL": "RBP", "BP": "RBP", "EBP": "RBP", "RBP": "RBP",
            "SPL": "RSP", "SP": "RSP", "ESP": "RSP", "RSP": "RSP",
            "RIP": "RIP",
        }
        for i in range(8, 16):
            r = f"R{i}"
            sub_to_full[r] = r
            sub_to_full[f"R{i}B"] = r
            sub_to_full[f"R{i}W"] = r
            sub_to_full[f"R{i}D"] = r
        # Iterative propagation (up to 20 passes — sufficient for convergence in small functions)
        state = defaultdict(lambda: "unknown")
        self.seed_initial_state(state, func, sub_to_full)
        for _ in range(20):
            new_state = state.copy()
            for bb in func.basic_blocks:
                for instr in bb.instructions:
                    self.transfer(instr, state, new_state, sub_to_full, func)
            if new_state == state:
                break
            state = new_state
        # Final annotation pass
        state = defaultdict(lambda: "unknown")
        self.seed_initial_state(state, func, sub_to_full)
        for bb in func.basic_blocks:
            for instr in bb.instructions:
                self.transfer(instr, state, state, sub_to_full, func, annotate=True)
    def seed_initial_state(self, state: Dict[str, str], func: Function, sub_to_full: dict):
        state["RSP"] = "ptr"
        if func.uses_frame_pointer:
            state["RBP"] = "ptr"
        for arg in func.arguments:
            if arg.kind == "register":
                full = sub_to_full.get(arg.location, arg.location)
                if "*" in str(arg.inferred_type):
                    state[full] = "ptr"
                elif arg.inferred_type == "i32":
                    state[full] = "i32_unsigned"
                else:
                    state[full] = "i64"
        # Stack slots (rare in sample, but supported)
        for slot in func.stack_slots:
            key = f"stack_{slot.offset}"
            state[key] = "i64" # conservative default
    def transfer(self, instr: Instruction, state: Dict[str, str], new_state: Dict[str, str], sub_to_full: dict, func: Function, annotate: bool = False):
        opcode = instr.opcode.replace("LOCK ", "")
        # 1. Identify conversion-to-integer ops (Fix for CVTTSS2SI)
        # CVTTSS2SI, CVTTSD2SI, CVTSS2SI, CVTSD2SI produce signed integers
        is_cvt_to_int = opcode in ("CVTTSS2SI", "CVTTSD2SI", "CVTSS2SI", "CVTSD2SI")
        # Determine operation characteristics
        # Fix 1: Exclude conversions from float ops so they produce int types
        is_float_op = ("SS" in opcode or "SD" in opcode or opcode in ("MOVSS", "MOVSD")) and not is_cvt_to_int
        float_width = 32 if "SS" in opcode else 64 if "SD" in opcode else 0
        # Added CVT...SI to signed ops
        is_signed_op = opcode in ("IMUL", "IDIV", "MOVSX", "SAR", "CDQ", "CQO", "CVTTSS2SI", "CVTTSD2SI", "CVTSS2SI", "CVTSD2SI")
        # NEW: unsigned ops (zero-extension or flag-setting producing 0/1)
        is_unsigned_op = opcode == "MOVZX" or opcode.startswith("SET")
        op_width = 64
        if instr.operands:
            dest = instr.operands[0]
            if dest.size:
                op_width = SIZE_MAP.get(dest.size, 64)
            elif dest.register:
                op_width = get_reg_width(dest.register)
        # Default op_refinement
        op_ref = "unknown"
        if is_cvt_to_int:
            # Conversion to signed integer
            op_ref = f"i{op_width}_signed"
        elif is_float_op:
            op_ref = "f32" if float_width == 32 else "f64"
        else:
            # Integer case – distinguish signed/unsigned/neutral
            if is_signed_op:
                op_ref = f"i{op_width}_signed"
            elif is_unsigned_op:
                op_ref = f"i{op_width}_unsigned"
            else:
                op_ref = f"i{op_width}"
        # Special overrides for CDQ/CQO to reflect their operand width
        if opcode == "CDQ":
            op_ref = "i32_signed"
        elif opcode == "CQO":
            op_ref = "i64_signed"
        # Detect likely pointer loads: 64-bit load from pointer base (simple addressing)
        if opcode == "MOV" and len(instr.operands) == 2 and instr.operands[0].register and instr.operands[1].memory and op_width == 64:
            mem = instr.operands[1].memory
            if mem.base and not mem.index and (mem.scale is None or mem.scale == 1):
                full_base = sub_to_full.get(mem.base, mem.base)
                if state[full_base] == "ptr":
                    op_ref = "ptr"
        # Fix 3: Pointer vs. Integer distinction for MOV/XCHG/CMOVE
        # If we move a known pointer value, the operation refinement should be "ptr"
        if opcode in ("MOV", "XCHG", "CMOVE") and len(instr.operands) == 2:
            src = instr.operands[1]
            if src.register:
                src_full = sub_to_full.get(src.register, src.register)
                if state.get(src_full) == "ptr":
                    op_ref = "ptr"
        # Special: LEA pointer vs arithmetic detection (enhanced)
        if opcode == "LEA" and len(instr.operands) == 2:
            src = instr.operands[1]
            op_ref = "ptr" # default: assume pointer-producing
            if src.memory:
                mem = src.memory
                is_arith = True
                # RIP-relative is always a pointer-producing address
                if mem.base == "RIP":
                    is_arith = False
                if mem.base:
                    full = sub_to_full.get(mem.base, mem.base)
                    if state[full] == "ptr":
                        is_arith = False
                if mem.index:
                    full = sub_to_full.get(mem.index, mem.index)
                    if state[full] == "ptr":
                        is_arith = False
                if is_arith:
                    op_ref = f"i{op_width}"
            elif getattr(src, "expression", None) is not None:
                # dict expressions are arithmetic (e.g., additive); str are symbols -> pointer
                if isinstance(src.expression, dict):
                    op_ref = f"i{op_width}"
                # else: str symbol/label -> keep "ptr"
        # Special for control-flow instructions
        is_control_flow = opcode.startswith("J") or opcode in ("CALL", "RET", "LOOP")
        if is_control_flow:
            op_ref = None
        # Preserve pointer semantics for common pointer arithmetic (prevent downgrade to plain integer)
        if op_ref != "ptr": # Do not override if already identified as ptr (e.g. control flow)
            if opcode in ("INC", "DEC"):
                if instr.operands:
                    dest = instr.operands[0]
                    if dest.register:
                        full = sub_to_full.get(dest.register, dest.register)
                        if state.get(full, "unknown") == "ptr":
                            op_ref = "ptr"
            elif opcode in ("ADD", "SUB") and len(instr.operands) == 2:
                src = instr.operands[1]
                if getattr(src, "integer", None) is not None: # immediate offset
                    dest = instr.operands[0]
                    if dest.register:
                        full = sub_to_full.get(dest.register, dest.register)
                        if state.get(full, "unknown") == "ptr":
                            op_ref = "ptr"
        if annotate:
            if op_ref is not None:
                instr.op_refinement = op_ref
        # Annotate operand uses
        if annotate:
            for op in instr.operands:
                if op.register:
                    full = sub_to_full.get(op.register, op.register)
                    op.inferred_type = state.get(full, "unknown")
                if op.memory:
                    op.address_refinement = "ptr"
                    # Conservative pointee from op_width
                    op.inferred_type = f"i{op_width}" if not is_float_op else op_ref
                elif op.name or getattr(op, "symbol_ref", None) or getattr(op, "expression", None):
                    op.inferred_type = "ptr"
        # Defs / updates
        result_ref = op_ref if op_ref is not None else "unknown"
        if instr.operands:
            dest = instr.operands[0]
            if dest.register:
                full = sub_to_full.get(dest.register, dest.register)
                if is_zeroing_write(dest.register):
                    result_ref = "i64_unsigned" # zero-extension forces unsigned promotion
                new_state[full] = lub(new_state[full], result_ref)
                # Propagate type on direct register-to-register MOV (helps pointer flow, e.g., frame pointer)
                if opcode == "MOV" and len(instr.operands) == 2 and instr.operands[1].register:
                    src_full = sub_to_full.get(instr.operands[1].register, instr.operands[1].register)
                    src_type = state[src_full]
                    new_state[full] = lub(new_state[full], src_type)
            # Special extensions
            if opcode == "MOVSX":
                result_ref = "i64_signed"
                if dest.register:
                    full = sub_to_full.get(dest.register, dest.register)
                    new_state[full] = lub(new_state[full], result_ref)
            elif opcode == "MOVZX":
                result_ref = "i64_unsigned"
                if dest.register:
                    full = sub_to_full.get(dest.register, dest.register)
                    new_state[full] = lub(new_state[full], result_ref)
            elif opcode == "CDQ":
                if state["RAX"].endswith("signed"):
                    new_state["RDX"] = "i32_signed"
            elif opcode == "CQO":
                if state["RAX"].endswith("signed"):
                    new_state["RDX"] = "i64_signed"
        # SSE scalar
        if is_float_op:
            for op in instr.operands:
                if op.register and op.register.startswith("XMM"):
                    new_state[op.register] = op_ref
        # Stack slots
        for slot in func.stack_slots:
            key = f"stack_{slot.offset}"
            for op in instr.operands:
                if op.memory and isinstance(op.memory.displacement, int) and op.memory.displacement == slot.offset:
                    if len(instr.operands) > 1 and instr.operands[1].register: # store
                        full = sub_to_full.get(instr.operands[1].register, instr.operands[1].register)
                        r = state.get(full, "unknown")
                        new_state[key] = lub(new_state[key], r)
                        if annotate:
                            op.inferred_type = r
                    else: # load
                        r = state.get(key, "unknown")
                        if annotate:
                            op.inferred_type = r
                        if instr.operands[0].register:
                            full = sub_to_full.get(instr.operands[0].register, instr.operands[0].register)
                            new_state[full] = lub(new_state[full], r)
def main():
    try:
        raw = sys.stdin.read()
        obj = json.loads(raw)
        ast = legacy_program_dict_to_ast(obj, include_enhancements=True)
        Step8Refinement(ast).run()
        out = ast_to_legacy_program_dict(ast, include_enhancements=True)
        json.dump(out, sys.stdout, indent=2)
        print()
    except Exception as e:
        sys.stderr.write(f"Step 8 failed: {e}\n")
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
if __name__ == "__main__":
    main()
