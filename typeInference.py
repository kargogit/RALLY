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
from typing import Dict, Set, List, Any, Optional
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
    return width == 32  # only 32-bit writes zero the upper 32 bits of the 64-bit register (x86-64 semantics)
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
        # Per-BB State Maps: bb_id -> { reg: type } / { reg: set(labels) }
        bb_in_type: Dict[str, Dict[str, str]] = {}
        bb_out_type: Dict[str, Dict[str, str]] = {}
        bb_in_block: Dict[str, Dict[str, Set[str]]] = {}
        bb_out_block: Dict[str, Dict[str, Set[str]]] = {}
        # Initialize states
        seeds = self.seed_initial_state(func, sub_to_full)
        for bb in func.basic_blocks:
            bid = bb.id if bb.id else f"bb_{id(bb)}"
            # Entry BB gets seeds, others start unknown/empty
            if not bb.predecessors:
                bb_in_type[bid] = seeds.copy()
                bb_in_block[bid] = defaultdict(set)
            else:
                bb_in_type[bid] = defaultdict(lambda: "unknown")
                bb_in_block[bid] = defaultdict(set)
            # Initialize out as copy of in (will be overwritten in loop)
            bb_out_type[bid] = bb_in_type[bid].copy()
            bb_out_block[bid] = {k: set(v) for k, v in bb_in_block[bid].items()}
        # Fixed-point iteration
        for _ in range(50): # Increased limit for safety
            changed = False
            for bb in func.basic_blocks:
                bid = bb.id if bb.id else f"bb_{id(bb)}"
                # Compute In-State from Predecessors' Out-States
                if bb.predecessors:
                    new_in_type: Dict[str, str] = defaultdict(lambda: "unknown")
                    new_in_block: Dict[str, Set[str]] = defaultdict(set)
                    for pred in bb.predecessors:
                        pid = pred.id if pred.id else f"bb_{id(pred)}"
                        pred_out_t = bb_out_type.get(pid, {})
                        pred_out_b = bb_out_block.get(pid, {})
                        # LUB Types
                        for reg, t in pred_out_t.items():
                            new_in_type[reg] = lub(new_in_type.get(reg, "unknown"), t)
                        # Union Block Addresses
                        for reg, labels in pred_out_b.items():
                            new_in_block[reg] |= labels
                    # Check for changes in In-State
                    # Note: We compare new_in with stored bb_in_type
                    # For types
                    for k in set(new_in_type.keys()) | set(bb_in_type[bid].keys()):
                        if new_in_type.get(k, "unknown") != bb_in_type[bid].get(k, "unknown"):
                            changed = True
                            bb_in_type[bid][k] = new_in_type.get(k, "unknown")
                    # For blocks
                    for k in set(new_in_block.keys()) | set(bb_in_block[bid].keys()):
                        if new_in_block.get(k, set()) != bb_in_block[bid].get(k, set()):
                            changed = True
                            bb_in_block[bid][k] = set(new_in_block.get(k, set()))
                else:
                    # Entry block: ensure seeds are present
                    for k, v in seeds.items():
                        if bb_in_type[bid].get(k, "unknown") != v:
                            bb_in_type[bid][k] = v
                            changed = True
                # Compute Out-State by processing instructions
                # Start with a copy of In-State
                current_type = bb_in_type[bid].copy()
                current_block = {k: set(v) for k, v in bb_in_block[bid].items()}
                for instr in bb.instructions:
                    self.transfer(instr, current_type, current_block, sub_to_full, func, annotate=False)
                # Update stored Out-State
                # Check for changes
                if current_type != bb_out_type[bid]:
                    changed = True
                    bb_out_type[bid] = current_type
                if current_block != bb_out_block[bid]:
                    changed = True
                    bb_out_block[bid] = current_block
            if not changed:
                break
        # Post-convergence: resolve indirect targets, update CFG, finalize noreturn
        self.resolve_indirects(func, bb_out_block, label_to_bb)
        # Final annotation pass using converged states
        for bb in func.basic_blocks:
            bid = bb.id if bb.id else f"bb_{id(bb)}"
            # Start with In-State for this BB
            current_type = bb_in_type[bid].copy()
            current_block = {k: set(v) for k, v in bb_in_block[bid].items()}
            for instr in bb.instructions:
                self.transfer(instr, current_type, current_block, sub_to_full, func, annotate=True)
    def seed_initial_state(self, func: Function, sub_to_full: dict) -> Dict[str, str]:
        seeds: Dict[str, str] = {}
        seeds["RSP"] = "ptr"
        if func.uses_frame_pointer:
            seeds["RBP"] = "ptr"
        for arg in func.arguments:
            if arg.kind == "register":
                full = sub_to_full.get(arg.location, arg.location)
                if "*" in str(arg.inferred_type):
                    seeds[full] = "ptr"
                elif arg.inferred_type == "i32":
                    seeds[full] = "i32_unsigned"
                else:
                    seeds[full] = "i64"
        for slot in func.stack_slots:
            key = f"stack_{slot.offset}"
            seeds[key] = "i64"
        return seeds
    def transfer(self, instr: Instruction, type_state: Dict[str, str], block_state: Dict[str, Set[str]], sub_to_full: dict, func: Function, annotate: bool = False):
        opcode = instr.opcode.replace("LOCK ", "")
        is_cvt_to_int = opcode in ("CVTTSS2SI", "CVTTSD2SI", "CVTSS2SI", "CVTSD2SI")
        is_float_op = ("SS" in opcode or "SD" in opcode or opcode in ("MOVSS", "MOVSD")) and not is_cvt_to_int
        float_width = 32 if "SS" in opcode else 64 if "SD" in opcode else 0
        is_signed_op = opcode in ("IMUL", "IDIV", "MOVSX", "SAR", "CDQ", "CQO", "CDQE", "CWDE", "CVTTSS2SI", "CVTTSD2SI", "CVTSS2SI", "CVTSD2SI")
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
        elif opcode == "CDQE":
            op_ref = "i64_signed"
        elif opcode == "CWDE":
            op_ref = "i32_signed"
        if opcode == "MOV" and len(instr.operands) == 2 and instr.operands[0].register and instr.operands[1].memory and op_width == 64:
            mem = instr.operands[1].memory
            if mem.base and not mem.index and (mem.scale is None or mem.scale == 1):
                full_base = sub_to_full.get(mem.base, mem.base)
                if type_state.get(full_base) == "ptr":
                    op_ref = "ptr"
        if opcode in ("MOV", "XCHG", "CMOVE") and len(instr.operands) == 2:
            src = instr.operands[1]
            if src.register:
                src_full = sub_to_full.get(src.register, src.register)
                if type_state.get(src_full) == "ptr":
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
                    if type_state.get(full) == "ptr":
                        is_arith = False
                if mem.index:
                    full = sub_to_full.get(mem.index, mem.index)
                    if type_state.get(full) == "ptr":
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
                        if type_state.get(full, "unknown") == "ptr":
                            op_ref = "ptr"
            elif opcode in ("ADD", "SUB") and len(instr.operands) == 2:
                src = instr.operands[1]
                if getattr(src, "integer", None) is not None:
                    dest = instr.operands[0]
                    if dest.register:
                        full = sub_to_full.get(dest.register, dest.register)
                        if type_state.get(full, "unknown") == "ptr":
                            op_ref = "ptr"
        # Address computation refinement (meets spec: participating address registers are refined toward pointer-like usage)
        # Value refinement remains based on access width/opcode; this only affects the address side and is conservative.
        for op in instr.operands:
            if getattr(op, "memory", None) is not None:
                mem = op.memory
                for reg_field in ("base", "index"):
                    r = getattr(mem, reg_field, None)
                    if r:
                        full = sub_to_full.get(r, r)
                        curr_t = type_state.get(full, "unknown")
                        if curr_t.startswith("i") or curr_t == "unknown":
                            type_state[full] = "ptr"
        if annotate:
            if op_ref is not None:
                instr.op_refinement = op_ref
            for op in instr.operands:
                if op.register:
                    full = sub_to_full.get(op.register, op.register)
                    op.inferred_type = type_state.get(full, "unknown")
                if op.memory:
                    op.address_refinement = "ptr"
                    op.inferred_type = f"i{op_width}" if not is_float_op else (op_ref if op_ref else "unknown")
                elif op.name or getattr(op, "symbol_ref", None) or getattr(op, "expression", None):
                    op.inferred_type = "ptr"
        # Update State (Definition vs Refinement)
        # For definitions (MOV, LEA, etc.), we assign. For others, we could LUB,
        # but since we walk sequentially, assignment of the computed result_ref is usually correct
        # provided result_ref is not 'unknown'.
        if instr.operands:
            dest = instr.operands[0]
            if dest.register:
                full = sub_to_full.get(dest.register, dest.register)
                result_ref = op_ref if op_ref is not None else "unknown"
                if is_zeroing_write(dest.register):
                    result_ref = "i64_unsigned"
                if result_ref != "unknown" and get_reg_width(dest.register) in (32, 64):
                    type_state[full] = result_ref
                # Special handling for MOV source type propagation
                if opcode == "MOV" and len(instr.operands) == 2 and instr.operands[1].register:
                    src_full = sub_to_full.get(instr.operands[1].register, instr.operands[1].register)
                    src_type = type_state.get(src_full, "unknown")
                    if src_type != "unknown":
                        # MOV propagates source type if dest ref is generic
                        if result_ref == "unknown" or result_ref.startswith("i64"):
                             # Keep specific source type if dest is wide
                             type_state[full] = src_type
                if opcode == "MOVSX":
                    type_state[full] = "i64_signed"
                elif opcode == "MOVZX":
                    type_state[full] = "i64_unsigned"
                elif opcode == "CDQ":
                    if type_state.get("RAX", "").endswith("signed"):
                        type_state["RDX"] = "i32_signed"
                elif opcode == "CQO":
                    if type_state.get("RAX", "").endswith("signed"):
                        type_state["RDX"] = "i64_signed"
                elif opcode == "CDQE":
                    type_state["RAX"] = "i64_signed"
                elif opcode == "CWDE":
                    type_state["RAX"] = "i32_signed"
        if is_float_op:
            for op in instr.operands:
                if op.register and op.register.startswith("XMM"):
                    type_state[op.register] = op_ref
        # Stack Slots
        for slot in func.stack_slots:
            key = f"stack_{slot.offset}"
            for op in instr.operands:
                if op.memory and isinstance(op.memory.displacement, int) and op.memory.displacement == slot.offset:
                    if len(instr.operands) > 1 and instr.operands[1].register:
                        full = sub_to_full.get(instr.operands[1].register, instr.operands[1].register)
                        r = type_state.get(full, "unknown")
                        if r != "unknown":
                            type_state[key] = r # Store refines slot
                        if annotate:
                            op.inferred_type = r
                    else:
                        r = type_state.get(key, "unknown")
                        if annotate:
                            op.inferred_type = r
                        if instr.operands and instr.operands[0].register:
                            full = sub_to_full.get(instr.operands[0].register, instr.operands[0].register)
                            if r != "unknown":
                                type_state[full] = r # Load refines reg
        # Block-address transfer
        if instr.operands and instr.operands[0].register:
            dest = instr.operands[0]
            full_dest = sub_to_full.get(dest.register, dest.register)
            if full_dest not in block_state:
                block_state[full_dest] = set()
            if opcode == "LEA" and len(instr.operands) == 2:
                src = instr.operands[1]
                symbol_ref = getattr(src, "symbol_ref", None)
                if symbol_ref:
                    label_name = symbol_ref.name if hasattr(symbol_ref, "name") else (symbol_ref if isinstance(symbol_ref, str) else None)
                    if isinstance(label_name, str) and label_name in self.program.symbol_table:
                        info = self.program.symbol_table[label_name]
                        if info.get("kind") == "label" and info.get("definition", {}).get("bb_id"):
                            block_state[full_dest] = {label_name}
            elif opcode in ("MOV", "CMOVE") and len(instr.operands) == 2 and instr.operands[1].register:
                src = instr.operands[1]
                full_src = sub_to_full.get(src.register, src.register)
                src_blocks = block_state.get(full_src, set())
                if opcode == "CMOVE":
                    block_state[full_dest] |= src_blocks
                else:
                    block_state[full_dest] = src_blocks.copy()
            elif opcode in ("ADD", "SUB", "XOR", "AND", "OR", "IMUL", "SHL", "SAR", "INC", "DEC"):
                block_state[full_dest] = set()
            elif opcode in ("MOV", "MOVZX", "MOVSX") and len(instr.operands) == 2 and instr.operands[1].memory:
                block_state[full_dest] = set()
    def resolve_indirects(self, func: Function, bb_out_block: Dict[str, Dict[str, Set[str]]], label_to_bb: Dict[str, BasicBlock]):
        """Post-convergence indirect target resolution + CFG finalization."""
        for bb in func.basic_blocks:
            if not bb.instructions:
                continue
            term = bb.instructions[-1]
            if getattr(term, 'indirect_jump_kind', None) != "intraprocedural":
                continue
            if not term.operands or not term.operands[0].register:
                continue
            reg = term.operands[0].register
            bid = bb.id if bb.id else f"bb_{id(bb)}"
            # Use the converged OUT state of this specific BB
            targets_set = bb_out_block.get(bid, {}).get(reg, set())
            resolved = [label_to_bb[lbl] for lbl in targets_set if lbl in label_to_bb]
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
        # Prune fall-through after calls to noreturn functions
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
                if bb.successors:
                    for succ in list(bb.successors):
                        if getattr(succ, 'predecessors', None) and bb in succ.predecessors:
                            succ.predecessors.remove(bb)
                    bb.successors.clear()
        # Noreturn finalization
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
