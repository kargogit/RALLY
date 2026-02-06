## step5_cfg_refiner.py
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
# Known external noreturn functions
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

def get_function_containing_bb(bb: BasicBlock, program: Program) -> Optional[Function]:
    """Finds the parent Function of a BasicBlock."""
    # Check parent reference first if populated by Step 4
    if bb.parent and isinstance(bb.parent, Function):
        return bb.parent
    
    # Fallback search
    for sec in program.sections:
        for child in sec.children:
            if isinstance(child, Function):
                if bb in child.basic_blocks:
                    return child
    return None

def get_call_target_name(instr: Instruction) -> Optional[str]:
    """
    Extracts the target function name from a CALL instruction's operands.
    Checks 'name' field or 'symbol_ref'.
    """
    if not instr.operands:
        return None
    # Assuming the first operand of CALL is the target
    op = instr.operands[0]
    if op.name:
        return op.name
    if op.symbol_ref and isinstance(op.symbol_ref, Label):
        return op.symbol_ref.name
    return None

def resolve_target_function(instr: Instruction, func_map: Dict[str, Function]) -> Optional[Function]:
    """
    Resolves the target Function object for a CALL instruction if it is internal
    and direct (i.e., target_blocks is populated).
    """
    if not instr.target_blocks:
        return None
    
    # target_blocks contains BasicBlocks. The entry block of a function is the target.
    target_bb = instr.target_blocks[0]
    if not target_bb:
        return None
        
    # Find the function containing this target basic block
    for func in func_map.values():
        if target_bb in func.basic_blocks:
            return func
    return None

# ---- Core Analysis Logic -------------------------------------------------

def analyze_noreturn_behavior(program: Program):
    """
    Detects noreturn functions and updates CFG edges.
    """
    # 1. Build mappings
    func_map: Dict[str, Function] = {}
    bb_map: Dict[str, BasicBlock] = {} # bb_id -> bb
    
    for sec in program.sections:
        for child in sec.children:
            if isinstance(child, Function):
                func_map[child.id] = child
                for bb in child.basic_blocks:
                    bb_map[bb.id] = bb

    # 2. Initialize Noreturn Status
    # status: func_id (or external name) -> bool (True if noreturn)
    noreturn_status: Dict[str, bool] = {}
    
    # Initialize external known noreturns
    for ext_name in EXTERNAL_NORETURN_FUNCS:
        noreturn_status[ext_name] = True
        
    # Initialize internal functions conservatively (False = returns)
    for func_id in func_map:
        noreturn_status[func_id] = False

    # 3. Worklist Algorithm for Noreturn Detection
    worklist: Deque[str] = deque(func_map.keys())
    
    # Cache of function entry block IDs
    func_entry_bb_id: Dict[str, str] = {}
    for fid, func in func_map.items():
        if func.basic_blocks:
            func_entry_bb_id[fid] = func.basic_blocks[0].id

    while worklist:
        fid = worklist.popleft()
        func = func_map[fid]
        
        # Check if function is noreturn
        # A function is noreturn if no RET instruction is reachable from entry.
        
        is_currently_noreturn = noreturn_status[fid]
        
        # Reachability analysis (BFS)
        visited: Set[str] = set()
        queue: Deque[str] = deque()
        
        entry_id = func_entry_bb_id.get(fid)
        if not entry_id:
            continue # Empty function
            
        queue.append(entry_id)
        visited.add(entry_id)
        
        found_ret = False
        
        while queue:
            bb_id = queue.popleft()
            bb = bb_map.get(bb_id)
            if not bb: continue
            
            # Check terminator
            term = bb.terminator
            if term:
                opc = term.opcode.upper()
                
                if is_return_opcode(opc):
                    found_ret = True
                    break
                    
                elif is_call_opcode(opc):
                    # Determine if this call returns
                    call_returns = True # Conservative default
                    
                    target_func = resolve_target_function(term, func_map)
                    if target_func:
                        # Internal direct call
                        if noreturn_status.get(target_func.id, False):
                            call_returns = False
                    else:
                        # External or Indirect
                        name = get_call_target_name(term)
                        if name and noreturn_status.get(name, False):
                            call_returns = False
                    
                    if call_returns:
                        # Assume fall-through
                        # Find next block in linear order
                        bb_list = func.basic_blocks
                        try:
                            current_idx = bb_list.index(bb)
                            if current_idx + 1 < len(bb_list):
                                next_bb = bb_list[current_idx + 1]
                                if next_bb.id not in visited:
                                    visited.add(next_bb.id)
                                    queue.append(next_bb.id)
                        except ValueError:
                            pass # Should not happen
                
                elif is_jmp_opcode(opc):
                    # Check for tail call (jmp to function entry)
                    target_func = resolve_target_function(term, func_map)
                    if target_func:
                        # Jump to function entry.
                        # The path returns iff the target function returns.
                        if not noreturn_status.get(target_func.id, False):
                            # Target returns, so we effectively return here?
                            # Technically control flow enters the other function.
                            # If that function has a RET, this path leads to a RET.
                            # Simplification: If we JMP to a returning function, we don't execute a RET *here*,
                            # but the behavior is "returning". 
                            # The definition of "noreturn" for 'func' is "func never returns to caller".
                            # If func jumps to another function that returns, 'func' returns.
                            found_ret = True
                            break
                        # Else: Jump to noreturn function, so this path ends (no ret)
                    
                    # Standard intra-procedural jumps are handled by successors
                    pass 

            # Add explicit successors
            for succ in bb.successors:
                if succ.id not in visited:
                    # Only add if we haven't determined this path stops
                    # (e.g. if we hit a noreturn call above, we didn't add fallthrough, 
                    # but we might have other branches)
                    visited.add(succ.id)
                    queue.append(succ.id)
        
        # Update status if changed
        is_definitely_noreturn = not found_ret
        if is_definitely_noreturn != is_currently_noreturn:
            noreturn_status[fid] = is_definitely_noreturn
            # Re-queue all functions that call this one (interprocedural propagation)
            # Since we don't have a full call graph, we just re-queue everyone
            # or scan. Re-queueing everyone is safer and reasonably fast for small/med programs.
            for other_fid in func_map:
                if other_fid != fid:
                    worklist.append(other_fid)

    # 4. Annotate Functions
    for fid, func in func_map.items():
        if noreturn_status[fid]:
            func.noreturn_kind = "noreturn"
        else:
            func.noreturn_kind = None # Explicitly clear if needed

    # 5. Prune Spurious Fall-Through Edges
    for func in func_map.values():
        bb_list = func.basic_blocks
        for i, bb in enumerate(bb_list):
            term = bb.terminator
            if term and is_call_opcode(term.opcode):
                # Determine if this call is noreturn
                is_noreturn_call = False
                
                target_func = resolve_target_function(term, func_map)
                if target_func:
                    if noreturn_status.get(target_func.id, False):
                        is_noreturn_call = True
                else:
                    name = get_call_target_name(term)
                    if name and noreturn_status.get(name, False):
                        is_noreturn_call = True
                
                if is_noreturn_call:
                    # Identify fall-through block (next in list)
                    if i + 1 < len(bb_list):
                        fallthrough_bb = bb_list[i + 1]
                        
                        # Remove from successors
                        # We must be careful to compare by ID or object identity to avoid "TypeError: unhashable type"
                        if fallthrough_bb in bb.successors:
                            bb.successors.remove(fallthrough_bb)
                        
                        # Remove from predecessors
                        if bb in fallthrough_bb.predecessors:
                            fallthrough_bb.predecessors.remove(bb)

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
        # Deserialize with enhancements enabled to get parent links etc.
        ast_prog: Program = legacy_program_dict_to_ast(obj, include_enhancements=True)
        
        # Perform Step 5 Refinement
        refined_prog = analyze_noreturn_behavior(ast_prog)
        
        # Serialize back
        legacy_out = ast_to_legacy_program_dict(refined_prog, include_instr_locations=False, include_enhancements=True)
        json.dump(legacy_out, sys.stdout, indent=2, sort_keys=False)
        sys.stdout.write("\n")
        
    except Exception as exc:
        sys.stderr.write(f"Unexpected error in Step 5: {repr(exc)}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
