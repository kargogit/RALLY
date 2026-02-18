"""
typeInference.py
Step 8: Intraprocedural Data-Flow Analysis for Type Refinement (Revised for block-address tracking).
Performs lightweight, iterative fixed-point data-flow analysis over function basic blocks
to propagate and refine type information for registers (including subregisters), scalar
operations, operands, memory accesses, and stack slots **plus block-address provenance**.
Infers and annotates:
- Operation refinements (op_refinement) on instructions: integer width/signedness/neutral,
  pointer vs. arithmetic distinction, basic scalar floating-point (f32/f64).
- Operand types (inferred_type): detailed scalar integer types, signedness, pointers,
  and floating-point.
- **Block address sets for indirect jumps (jmp reg, call reg) via LEA to code labels,
  MOV/CMOVE propagation, and conservative clearing on destructive ops**.
- **Post-convergence: fully populates 'indirect_targets' (exact or small-func over-approx),
  updates bidirectional CFG edges, finalizes noreturn for pending functions**.
Uses a least-upper-bound (LUB) lattice for type merging with conservative fallbacks.
Inputs: Enriched AST from Step 7 (via stdin JSON).
Outputs: Further enriched AST with type refinement annotations + resolved indirect metadata (via stdout JSON).
"""
import sys
import json
from collections import defaultdict
from typing import Dict, Set
from astNodes import (
    legacy_program_dict_to_ast,
    ast_to_legacy_program_dict,
    Program,
    Function,
    Instruction,
    BasicBlock,
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
    if reg.endswith("L") or reg.endswith("H") or reg.endswith("B"):
        return 8
    if len(reg) == 2 and reg.isalpha():
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
    if "ptr" in (a, b):
        if a == "ptr" and b == "ptr":
            return "ptr"
        return "i64"
    if a.startswith("f") or b.startswith("f"):
        return "unknown"
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

    def get_label_to_bb(self, func: Function) -> Dict[str, BasicBlock]:
        label_to_bb = {}
        for bb in func.basic_blocks:
            if bb.start_label:
                label_to_bb[bb.start_label.name] = bb
        return label_to_bb

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

        label_to_bb = self.get_label_to_bb(func)

        basic_state = defaultdict(lambda: "unknown")
        block_state: Dict[str, Set[str]] = defaultdict(set)

        self.seed_initial_state(basic_state, func, sub_to_full)

        # Fixed-point iteration (basic + block provenance)
        for _ in range(20):
            new_basic = basic_state.copy()
            new_block: Dict[str, Set[str]] = {k: set(v) for k, v in block_state.items()}
            changed = False
            for bb in func.basic_blocks:
                for instr in bb.instructions:
                    self.transfer(instr, basic_state, new_basic, block_state, new_block, sub_to_full, func, annotate=False)
            if new_basic == basic_state and all(sorted(new_block.get(k, [])) == sorted(block_state.get(k, [])) for k in set(new_block) | set(block_state)):
                break
            basic_state = new_basic
            block_state = new_block

        # Post-convergence: resolve indirect targets, update CFG, finalize noreturn (Revised Step 8)
        self.resolve_indirects(func, block_state, label_to_bb)

        # Final annotation pass (basic refinements only - block provenance not annotated)
        basic_state = defaultdict(lambda: "unknown")
        self.seed_initial_state(basic_state, func, sub_to_full)
        for bb in func.basic_blocks:
            for instr in bb.instructions:
                self.transfer(instr, basic_state, basic_state, block_state, {}, sub_to_full, func, annotate=True)

    def seed_initial_state(self, basic_state: Dict, func: Function, sub_to_full: dict):
        basic_state["RSP"] = "ptr"
        if func.uses_frame_pointer:
            basic_state["RBP"] = "ptr"
        for arg in func.arguments:
            if arg.kind == "register":
                full = sub_to_full.get(arg.location, arg.location)
                if "*" in str(arg.inferred_type):
                    basic_state[full] = "ptr"
                elif arg.inferred_type == "i32":
                    basic_state[full] = "i32_unsigned"
                else:
                    basic_state[full] = "i64"
        for slot in func.stack_slots:
            key = f"stack_{slot.offset}"
            basic_state[key] = "i64"

    def transfer(self, instr: Instruction, basic_state: Dict, new_basic: Dict, block_state: Dict, new_block: Dict, sub_to_full: dict, func: Function, annotate: bool = False):
        opcode = instr.opcode.replace("LOCK ", "")
        is_cvt_to_int = opcode in ("CVTTSS2SI", "CVTTSD2SI", "CVTSS2SI", "CVTSD2SI")
        is_float_op = ("SS" in opcode or "SD" in opcode or opcode in ("MOVSS", "MOVSD")) and not is_cvt_to_int
        float_width = 32 if "SS" in opcode else 64 if "SD" in opcode else 0
        is_signed_op = opcode in ("IMUL", "IDIV", "MOVSX", "SAR", "CDQ", "CQO", "CVTTSS2SI", "CVTTSD2SI", "CVTSS2SI", "CVTSD2SI")
        is_unsigned_op = opcode == "MOVZX" or opcode.startswith("SET")
        op_width = 64
        if instr.operands:
            dest = instr.operands[0]
            if dest.size:
                op_width = SIZE_MAP.get(dest.size, 64)
            elif dest.register:
                op_width = get_reg_width(dest.register)
        op_ref = "unknown"
        if is_cvt_to_int:
            op_ref = f"i{op_width}_signed"
        elif is_float_op:
            op_ref = "f32" if float_width == 32 else "f64"
        else:
            if is_signed_op:
                op_ref = f"i{op_width}_signed"
            elif is_unsigned_op:
                op_ref = f"i{op_width}_unsigned"
            else:
                op_ref = f"i{op_width}"
        if opcode == "CDQ":
            op_ref = "i32_signed"
        elif opcode == "CQO":
            op_ref = "i64_signed"
        if opcode == "MOV" and len(instr.operands) == 2 and instr.operands[0].register and instr.operands[1].memory and op_width == 64:
            mem = instr.operands[1].memory
            if mem.base and not mem.index and (mem.scale is None or mem.scale == 1):
                full_base = sub_to_full.get(mem.base, mem.base)
                if basic_state.get(full_base) == "ptr":
                    op_ref = "ptr"
        if opcode in ("MOV", "XCHG", "CMOVE") and len(instr.operands) == 2:
            src = instr.operands[1]
            if src.register:
                src_full = sub_to_full.get(src.register, src.register)
                if basic_state.get(src_full) == "ptr":
                    op_ref = "ptr"
        if opcode == "LEA" and len(instr.operands) == 2:
            src = instr.operands[1]
            op_ref = "ptr"
            if src.memory:
                mem = src.memory
                is_arith = True
                if mem.base == "RIP":
                    is_arith = False
                if mem.base:
                    full = sub_to_full.get(mem.base, mem.base)
                    if basic_state.get(full) == "ptr":
                        is_arith = False
                if mem.index:
                    full = sub_to_full.get(mem.index, mem.index)
                    if basic_state.get(full) == "ptr":
                        is_arith = False
                if is_arith:
                    op_ref = f"i{op_width}"
            elif getattr(src, "expression", None) is not None:
                if isinstance(src.expression, dict):
                    op_ref = f"i{op_width}"
        is_control_flow = opcode.startswith("J") or opcode in ("CALL", "RET", "LOOP")
        if is_control_flow:
            op_ref = None
        if op_ref != "ptr":
            if opcode in ("INC", "DEC"):
                if instr.operands:
                    dest = instr.operands[0]
                    if dest.register:
                        full = sub_to_full.get(dest.register, dest.register)
                        if basic_state.get(full, "unknown") == "ptr":
                            op_ref = "ptr"
            elif opcode in ("ADD", "SUB") and len(instr.operands) == 2:
                src = instr.operands[1]
                if getattr(src, "integer", None) is not None:
                    dest = instr.operands[0]
                    if dest.register:
                        full = sub_to_full.get(dest.register, dest.register)
                        if basic_state.get(full, "unknown") == "ptr":
                            op_ref = "ptr"
        if annotate:
            if op_ref is not None:
                instr.op_refinement = op_ref
            for op in instr.operands:
                if op.register:
                    full = sub_to_full.get(op.register, op.register)
                    op.inferred_type = basic_state.get(full, "unknown")
                if op.memory:
                    op.address_refinement = "ptr"
                    op.inferred_type = f"i{op_width}" if not is_float_op else (op_ref if op_ref else "unknown")
                elif op.name or getattr(op, "symbol_ref", None) or getattr(op, "expression", None):
                    op.inferred_type = "ptr"
        result_ref = op_ref if op_ref is not None else "unknown"
        if instr.operands:
            dest = instr.operands[0]
            if dest.register:
                full = sub_to_full.get(dest.register, dest.register)
                if is_zeroing_write(dest.register):
                    result_ref = "i64_unsigned"
                new_basic[full] = lub(new_basic.get(full, "unknown"), result_ref)
                if opcode == "MOV" and len(instr.operands) == 2 and instr.operands[1].register:
                    src_full = sub_to_full.get(instr.operands[1].register, instr.operands[1].register)
                    src_type = basic_state.get(src_full, "unknown")
                    new_basic[full] = lub(new_basic.get(full, "unknown"), src_type)
            if opcode == "MOVSX":
                result_ref = "i64_signed"
                if dest.register:
                    full = sub_to_full.get(dest.register, dest.register)
                    new_basic[full] = lub(new_basic.get(full, "unknown"), result_ref)
            elif opcode == "MOVZX":
                result_ref = "i64_unsigned"
                if dest.register:
                    full = sub_to_full.get(dest.register, dest.register)
                    new_basic[full] = lub(new_basic.get(full, "unknown"), result_ref)
            elif opcode == "CDQ":
                if basic_state.get("RAX", "").endswith("signed"):
                    new_basic["RDX"] = "i32_signed"
            elif opcode == "CQO":
                if basic_state.get("RAX", "").endswith("signed"):
                    new_basic["RDX"] = "i64_signed"
        if is_float_op:
            for op in instr.operands:
                if op.register and op.register.startswith("XMM"):
                    new_basic[op.register] = op_ref
        for slot in func.stack_slots:
            key = f"stack_{slot.offset}"
            for op in instr.operands:
                if op.memory and isinstance(op.memory.displacement, int) and op.memory.displacement == slot.offset:
                    if len(instr.operands) > 1 and instr.operands[1].register:
                        full = sub_to_full.get(instr.operands[1].register, instr.operands[1].register)
                        r = basic_state.get(full, "unknown")
                        new_basic[key] = lub(new_basic.get(key, "unknown"), r)
                        if annotate:
                            op.inferred_type = r
                    else:
                        r = basic_state.get(key, "unknown")
                        if annotate:
                            op.inferred_type = r
                        if instr.operands[0].register:
                            full = sub_to_full.get(instr.operands[0].register, instr.operands[0].register)
                            new_basic[full] = lub(new_basic.get(full, "unknown"), r)

        # NEW: Block-address transfer (Revised Step 8 - minimal, targeted)
        if instr.operands and instr.operands[0].register:
            dest = instr.operands[0]
            full_dest = sub_to_full.get(dest.register, dest.register)
            if full_dest not in new_block:
                new_block[full_dest] = set()
            if opcode == "LEA" and len(instr.operands) == 2:
                src = instr.operands[1]
                symbol_ref = getattr(src, "symbol_ref", None)
                if symbol_ref:
                    label_name = symbol_ref.name if hasattr(symbol_ref, "name") else (symbol_ref if isinstance(symbol_ref, str) else None)
                    if isinstance(label_name, str) and label_name in self.program.symbol_table:
                        info = self.program.symbol_table[label_name]
                        if info.get("kind") == "label" and info.get("definition", {}).get("bb_id"):
                            new_block[full_dest] = {label_name}
            elif opcode in ("MOV", "CMOVE") and len(instr.operands) == 2 and instr.operands[1].register:
                src = instr.operands[1]
                full_src = sub_to_full.get(src.register, src.register)
                src_blocks = block_state.get(full_src, set())
                if opcode == "CMOVE":
                    new_block[full_dest] |= src_blocks
                else:
                    new_block[full_dest] = src_blocks.copy()
            elif opcode in ("ADD", "SUB", "XOR", "AND", "OR", "IMUL", "SHL", "SAR", "INC", "DEC"):
                new_block[full_dest] = set()
            elif opcode in ("MOV", "MOVZX", "MOVSX") and len(instr.operands) == 2 and instr.operands[1].memory:
                new_block[full_dest] = set()  # memory load clears provenance (unless jump-table, omitted for practicality)

    def resolve_indirects(self, func: Function, block_state: Dict[str, Set[str]], label_to_bb: Dict[str, BasicBlock]):
        """Post-convergence indirect target resolution + CFG finalization (Revised Step 8)."""
        for bb in func.basic_blocks:
            if not bb.instructions:
                continue
            term = bb.instructions[-1]
            if getattr(term, 'indirect_jump_kind', None) != "intraprocedural":
                continue
            if not term.operands or not term.operands[0].register:
                continue
            reg = term.operands[0].register
            full = reg # R8 stays R8, etc.
            targets_set = block_state.get(full, set())
            resolved = [label_to_bb[lbl] for lbl in targets_set if lbl in label_to_bb]
            # Safe CFG update – NO set() on BasicBlock objects
            if bb.successors is None:
                bb.successors = []
            for tbb in resolved:
                if tbb not in bb.successors:
                    bb.successors.append(tbb)
                if tbb.predecessors is None:
                    tbb.predecessors = []
                if bb not in tbb.predecessors:
                    tbb.predecessors.append(bb)
            if resolved:
                term.indirect_targets = resolved
                term.indirect_targets_over_approximated = False
            elif len(func.basic_blocks) <= 10:
                all_bbs = list(func.basic_blocks)
                term.indirect_targets = all_bbs
                term.indirect_targets_over_approximated = True
                for tbb in all_bbs:
                    if tbb not in bb.successors:
                        bb.successors.append(tbb)
                    if tbb.predecessors is None:
                        tbb.predecessors = []
                    if bb not in tbb.predecessors:
                        tbb.predecessors.append(bb)

        # Prune fall-through after calls to noreturn functions (internal or exit)
        # This eliminates the spurious edges
        for bb in func.basic_blocks:
            if not bb.instructions:
                continue
            term = bb.instructions[-1]
            if term.opcode != "CALL" or not term.operands:
                continue
            callee_op = term.operands[0]
            callee_name = getattr(callee_op, 'name', None)
            if not callee_name:
                continue

            # Is the callee a noreturn function?
            is_noreturn_callee = (callee_name == "exit")
            if not is_noreturn_callee:
                for section in self.program.sections:
                    for child in section.children:
                        if isinstance(child, Function) and getattr(child, 'entry_label', None) == callee_name:
                            if getattr(child, 'noreturn_kind', None) == "noreturn":
                                is_noreturn_callee = True
                            break
                    if is_noreturn_callee:
                        break

            if is_noreturn_callee:
                # Remove fall-through successor(s) and keep bidirectional consistency
                if bb.successors:
                    for succ in list(bb.successors):
                        if getattr(succ, 'predecessors', None) and bb in succ.predecessors:
                            succ.predecessors.remove(bb)
                    bb.successors.clear()

        # Noreturn finalization for pending_resolution (unchanged)
        if getattr(func, 'noreturn_kind', None) == "pending_resolution":
            has_exit = any(
                i.opcode == "CALL" and i.operands and getattr(i.operands[0], 'name', None) == "exit"
                for bb_ in func.basic_blocks for i in bb_.instructions
            )
            if has_exit:
                func.noreturn_kind = "noreturn"

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
