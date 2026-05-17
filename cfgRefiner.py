# Refine the Control Flow Graph (CFG) by detecting noreturn functions
# and pruning spurious fall-through edges after calls to noreturn functions.
#
# Reads a legacy-program dict JSON from stdin (output of Step 4),
# deserializes to typed AST, performs analysis, and writes the
# updated legacy-program dict to stdout.

import sys
import json
from typing import Dict, List, Set, Optional, Deque
from collections import deque

# Import typed-ast helpers
from astNodes import (
    legacy_program_dict_to_ast,
    ast_to_legacy_program_dict,
    Program,
    Section,
    Function,
    BasicBlock,
    Instruction,
    Operand,
    Label,
)

# ---- Constants -----------------------------------------------------------
EXTERNAL_NORETURN_FUNCS = {
    'exit', '_exit', 'abort', '__stack_chk_fail',
    '_Exit', 'quick_exit', 'pthread_exit'
}

def is_call_opcode(opcode: str) -> bool:
    return opcode.upper() == 'CALL'

def is_return_opcode(opcode: str) -> bool:
    return opcode.upper() == 'RET'

def is_jmp_opcode(opcode: str) -> bool:
    return opcode.upper() == 'JMP'

# ---- Helper Functions ----------------------------------------------------
def is_unresolved_indirect_terminator(instr: Instruction) -> bool:
    if instr is None or is_return_opcode(instr.opcode):
        return False
    if len(instr.target_blocks) > 0 or len(instr.indirect_targets) > 0:
        return False
    if is_call_opcode(instr.opcode):
        if get_call_target_name(instr) is not None:
            return False
    return True

def has_unresolved_indirect_terminators(func: Function) -> bool:
    for bb in func.basic_blocks:
        if bb.terminator and is_unresolved_indirect_terminator(bb.terminator):
            return True
    return False

def get_function_containing_bb(bb: BasicBlock, program: Program) -> Optional[Function]:
    if bb.parent and isinstance(bb.parent, Function):
        return bb.parent

    bb_id = bb.id
    for sec in program.sections:
        for child in sec.children:
            if isinstance(child, Function):
                # --- FIX: compare by .id, not by __eq__ ---
                if any(b.id == bb_id for b in child.basic_blocks):
                    return child
    return None

def get_call_target_name(instr: Instruction) -> Optional[str]:
    if not instr.operands:
        return None
    op = instr.operands[0]
    if op.name:
        return op.name
    if op.symbol_ref and isinstance(op.symbol_ref, Label):
        return op.symbol_ref.name
    return None

def classify_call_target(instr: Instruction, func_map: Dict[str, Function]) -> Optional[str]:
    if not is_call_opcode(instr.opcode):
        return None
    if instr.target_blocks:
        return "internal"
    if get_call_target_name(instr) is not None:
        return "external"
    return "indirect"

def resolve_target_function_via_map(
    instr: Instruction,
    bb_id_to_func: Dict[str, Function],
) -> Optional[Function]:
    """
    Resolves the target Function for a CALL/JMP using a prebuilt
    bb_id → Function lookup.  Avoids __eq__ on BasicBlock entirely.
    """
    if not instr.target_blocks:
        return None
    target_bb = instr.target_blocks[0]
    if not target_bb:
        return None
    return bb_id_to_func.get(target_bb.id)

def is_noreturn_callee_target(
    instr: Instruction,
    noreturn_status: Dict[str, bool],
    bb_id_to_func: Dict[str, Function],
) -> bool:
    if not is_call_opcode(instr.opcode):
        return False
    target_func = resolve_target_function_via_map(instr, bb_id_to_func)
    if target_func:
        return noreturn_status.get(target_func.id, False)
    name = get_call_target_name(instr)
    if name:
        return noreturn_status.get(name, False)
    return False

# ---- Core Analysis Logic -------------------------------------------------
def analyze_noreturn_behavior(program: Program):
    # 1. Build mappings
    func_map: Dict[str, Function] = {}
    bb_map: Dict[str, BasicBlock] = {}
    # --- FIX: bb_id → owning Function (avoids __eq__ scan) ---
    bb_id_to_func: Dict[str, Function] = {}

    for sec in program.sections:
        for child in sec.children:
            if isinstance(child, Function):
                func_map[child.id] = child
                for bb in child.basic_blocks:
                    bb_map[bb.id] = bb
                    bb_id_to_func[bb.id] = child

    # Annotate call target classifications
    for func in func_map.values():
        for bb in func.basic_blocks:
            for instr in bb.instructions:
                instr.call_target_kind = classify_call_target(instr, func_map)

    # Detect functions with unresolved indirect terminators
    deferred_funcs: Set[str] = set()
    for fid, func in func_map.items():
        if has_unresolved_indirect_terminators(func):
            deferred_funcs.add(fid)

    # 2. Initialize Noreturn Status
    noreturn_status: Dict[str, bool] = {}
    for ext_name in EXTERNAL_NORETURN_FUNCS:
        noreturn_status[ext_name] = True
    for func_id in func_map:
        noreturn_status[func_id] = False

    # 3. Worklist Algorithm for Noreturn Detection
    worklist: Deque[str] = deque(
        fid for fid in func_map.keys() if fid not in deferred_funcs
    )
    func_entry_bb_id: Dict[str, str] = {}
    for fid, func in func_map.items():
        if func.basic_blocks:
            func_entry_bb_id[fid] = func.basic_blocks[0].id

    while worklist:
        fid = worklist.popleft()
        func = func_map[fid]
        is_currently_noreturn = noreturn_status[fid]

        visited: Set[str] = set()
        queue: Deque[str] = deque()
        entry_id = func_entry_bb_id.get(fid)
        if not entry_id:
            continue
        queue.append(entry_id)
        visited.add(entry_id)

        found_ret = False

        while queue:
            bb_id = queue.popleft()
            bb = bb_map.get(bb_id)
            if not bb:
                continue

            term = bb.terminator
            term_terminates_path = False

            if term:
                opc = term.opcode.upper()

                if is_return_opcode(opc):
                    found_ret = True
                    break

                elif is_call_opcode(opc):
                    # --- FIX: pass bb_id_to_func instead of func_map ---
                    if is_noreturn_callee_target(term, noreturn_status, bb_id_to_func):
                        term_terminates_path = True

                elif is_jmp_opcode(opc):
                    target_func = resolve_target_function_via_map(term, bb_id_to_func)
                    if target_func:
                        if not noreturn_status.get(target_func.id, False):
                            found_ret = True
                            break
                        else:
                            term_terminates_path = True

            if not term_terminates_path:
                for succ in bb.successors:
                    if succ.id not in visited:
                        visited.add(succ.id)
                        queue.append(succ.id)

        is_definitely_noreturn = not found_ret
        if is_definitely_noreturn != is_currently_noreturn:
            noreturn_status[fid] = is_definitely_noreturn
            for other_fid in func_map:
                if other_fid != fid and other_fid not in deferred_funcs:
                    worklist.append(other_fid)

    # 4. Annotate Functions
    for fid, func in func_map.items():
        if fid in deferred_funcs:
            func.noreturn_kind = "pending_resolution"
        elif noreturn_status.get(fid, False):
            func.noreturn_kind = "noreturn"
        else:
            func.noreturn_kind = None

    # 4b. Annotate call sites
    for func in func_map.values():
        for bb in func.basic_blocks:
            for instr in bb.instructions:
                if is_call_opcode(instr.opcode):
                    instr.is_noreturn_callee = is_noreturn_callee_target(
                        instr, noreturn_status, bb_id_to_func
                    )

    # 5. Prune Spurious Fall-Through Edges
    for func in func_map.values():
        if func.id in deferred_funcs:
            continue
        bb_list = func.basic_blocks
        for i, bb in enumerate(bb_list):
            term = bb.terminator
            if term and is_call_opcode(term.opcode):
                is_noreturn_call = is_noreturn_callee_target(
                    term, noreturn_status, bb_id_to_func
                )
                if is_noreturn_call:
                    if i + 1 < len(bb_list):
                        fallthrough_bb = bb_list[i + 1]
                        ft_id = fallthrough_bb.id
                        bb_id = bb.id

                        # --- FIX: filter by .id to avoid __eq__ recursion ---
                        bb.successors = [
                            s for s in bb.successors if s.id != ft_id
                        ]
                        fallthrough_bb.predecessors = [
                            p for p in fallthrough_bb.predecessors
                            if p.id != bb_id
                        ]
    return program

# ---- CLI Entry Point -----------------------------------------------------
def main():
    try:
        raw = sys.stdin.read()
        if not raw:
            sys.stderr.write("No input on stdin. Expecting JSON legacy program dict.\n")
            sys.exit(2)
        obj = json.loads(raw)
    except json.JSONDecodeError as jde:
        sys.stderr.write(f"Failed to parse JSON from stdin: {jde}\n")
        sys.exit(3)

    try:
        ast_prog: Program = legacy_program_dict_to_ast(obj, include_enhancements=True)
        refined_prog = analyze_noreturn_behavior(ast_prog)
        legacy_out = ast_to_legacy_program_dict(
            refined_prog, include_instr_locations=False, include_enhancements=True
        )
        json.dump(legacy_out, sys.stdout, indent=2, sort_keys=False)
        sys.stdout.write("\n")
    except Exception as exc:
        sys.stderr.write(f"Unexpected error in Step 5: {repr(exc)}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
