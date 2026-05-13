# Enhance typed-AST-based program with navigation, CFG links and symbol table.
# Reads a legacy-program dict JSON from stdin, deserializes to typed AST using
# legacy_program_dict_to_ast, mutates the AST to add enhancements, and writes
# the legacy-program dict (with enhancements) to stdout.
import sys
import json
from typing import Dict, List, Tuple, Any, Optional, Iterable
# Import typed-ast helpers
from astNodes import (
    legacy_program_dict_to_ast,
    ast_to_legacy_program_dict,
    Program,
    Section,
    Function,
    BasicBlock,
    Instruction,
    Label,
    Operand,
    Memory,
)
# ---- Utility helpers -----------------------------------------------------
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
def unique_id(prefix: str, counter: List[int]) -> str:
    counter[0] += 1
    return f"{prefix}{counter[0]}"
def iter_program_sections(program: Program) -> Iterable[Section]:
    for sec in program.sections:
        yield sec
def find_globals(program_legacy: Dict[str, Any]) -> List[str]:
    # this helper expects the raw legacy dict shape (we call it only on the raw input)
    items = program_legacy.get('program', [])
    if len(items) >= 2 and isinstance(items[1], dict) and 'globals' in items[1]:
        return items[1]['globals']
    return []
# ---- Opcode classification (simple, conservative) ------------------------
UNCONDITIONAL_JUMPS = {'JMP'}
CONDITIONAL_BRANCH_PREFIX = 'J' # JE, JNE, JL, JG, JO, JZ, etc (but not JMP)
CALL_OPS = {'CALL'}
RETURN_OPS = {'RET'}
LOOP_OPS = {'LOOP'}
HLT_OPS = {'HLT'}
def is_unconditional(opcode: str) -> bool:
    return opcode.upper() in UNCONDITIONAL_JUMPS
def is_conditional(opcode: str) -> bool:
    up = opcode.upper()
    return up.startswith(CONDITIONAL_BRANCH_PREFIX) and up not in UNCONDITIONAL_JUMPS
def is_call(opcode: str) -> bool:
    return opcode.upper() in CALL_OPS
def is_return(opcode: str) -> bool:
    return opcode.upper() in RETURN_OPS
def is_loop(opcode: str) -> bool:
    return opcode.upper() in LOOP_OPS
def is_hlt(opcode: str) -> bool:
    return opcode.upper() in HLT_OPS
# ---- Main enhancement pass -----------------------------------------------
def build_enhanced_program(legacy: Dict[str, Any]) -> Dict[str, Any]:
    # Deserialize into typed AST; request include_enhancements=True so we
    # pick up any pre-existing enhancement fields (they are stored in
    # _temp_* attributes and will be handled).
    ast_prog: Program = legacy_program_dict_to_ast(legacy, include_enhancements=True)
    # Establish parent ownership for sections to support complete upward navigation
    # (program → sections → functions → basic blocks → instructions) as required
    # for the CFG-aware representation.
    for sec in ast_prog.sections:
        sec.parent = ast_prog
    # Counters for stable IDs
    fid_ctr = [0]
    bb_ctr = [0]
    instr_ctr = [0]
    # Maps used during enhancement
    func_map: Dict[str, Function] = {}
    bb_map: Dict[str, BasicBlock] = {} # bb_id -> BasicBlock
    instr_map: Dict[str, Instruction] = {} # instr_id -> Instruction
    label_def_map: Dict[str, Dict[str, Any]] = {}
    data_symbol_map: Dict[str, Dict[str, Any]] = {}
    extern_symbols: set[str] = set()
    # gather globals from original legacy dict (so visibility decisions match)
    globals_list = find_globals(legacy)
    global_set = set(globals_list or [])
    # First pass: collect data/extern symbols from section pseudo_instructs,
    # and record label-groups (LabelGroup) label definitions.
    for sec in iter_program_sections(ast_prog):
        # pseudo_instruct entries are still raw dicts in Section.pseudo_instruct
        for pi in sec.pseudo_instruct or []:
            if isinstance(pi, dict):
                if 'name' in pi:
                    name = pi['name']
                    data_symbol_map[name] = {'kind': 'data', 'section': sec.name, 'def': pi}
                elif pi.get('directive') == 'extern':
                    params = pi.get('params', [])
                    if isinstance(params, list):
                        for p in params:
                            extern_symbols.add(p)
    # Process sections and functions: assign function IDs, bb IDs, instr IDs,
    # and normalize operands (RIP/GOT handling)
    for sec in ast_prog.sections:
        for child in sec.children:
            if isinstance(child, Function):
                func = child
                entry_label = func.entry_label
                if entry_label:
                    func_id = f"func:{entry_label}"
                else:
                    func_id = unique_id("func_", fid_ctr)
                func.id = func_id
                # ensure function.parent is the Section (serializer expects this)
                func.parent = sec
                func_map[func_id] = func
                # process basic blocks and instructions
                for bb in func.basic_blocks:
                    # assign id if missing
                    if not bb.id:
                        bb.id = unique_id("bb_", bb_ctr)
                    bb_map[bb.id] = bb
                    bb.parent = func
                    # if bb has a start_label object, record label_def_map
                    if bb.start_label:
                        label_def_map[bb.start_label.name] = {
                            'kind': 'label',
                            'defined_in': 'code',
                            'bb_id': bb.id,
                            'section': sec.name
                        }
                    # instructions: ensure ids and parents, and normalize operands
                    for instr in bb.instructions:
                        if not instr.id:
                            instr.id = unique_id("instr_", instr_ctr)
                        instr.parent = bb
                        instr_map[instr.id] = instr
                        # Normalize operands:
                        # 1. Convert LEA with bare label expression to RIP-relative memory (common under default rel)
                        # 2. Then handle existing RIP/GOT memory operands
                        for op in instr.operands:
                            if (instr.opcode.upper() == 'LEA' and
                                getattr(op, 'expression', None) is not None and
                                isinstance(op.expression, str)):
                                op.memory = Memory(base='RIP', displacement=op.expression)
                                op.expression = None
                            if isinstance(op, Operand) and op.memory is not None:
                                mem: Memory = op.memory
                                if isinstance(mem.base, str) and mem.base.upper() == 'RIP':
                                    op.rip_relative = True
                                    op.via_got = False
                                    disp = mem.displacement
                                    if isinstance(disp, str):
                                        low = disp.lower()
                                        if 'wrt' in low and 'got' in low:
                                            # split on 'wrt' similar to dict-based version
                                            sym = disp.split('wrt', 1)[0].strip()
                                            if sym:
                                                mem.displacement = sym
                                                op.via_got = True
    # Handle .init_array / .fini_array references (pseudo_instruct values)
    for sec in ast_prog.sections:
        for pi in sec.pseudo_instruct or []:
            for v in pi.get('values', []):
                if isinstance(v, dict) and 'symbol' in v:
                    sym = v['symbol']
                    if sym in label_def_map:
                        label_def_map[sym].setdefault('referenced_by', []).append({
                            'section': sec.name, 'pseudo': pi
                        })
    # label_to_bb mapping (for quick resolve)
    label_to_bb: Dict[str, str] = {lbl: info['bb_id'] for lbl, info in label_def_map.items() if 'bb_id' in info}
    # CFG construction (successors, terminators, target_blocks)
    for func_id, func in func_map.items():
        ordered_bbs = func.basic_blocks
        ordered_bb_ids = [bb.id for bb in ordered_bbs]
        bb_index = {bb_id: idx for idx, bb_id in enumerate(ordered_bb_ids)}
        for idx, bb in enumerate(ordered_bbs):
            last_instr = bb.instructions[-1] if bb.instructions else None
            def is_terminator_instr(instr: Optional[Instruction]) -> bool:
                if not instr:
                    return False
                opc = (instr.opcode or '').upper()
                if is_unconditional(opc) or is_conditional(opc) or is_return(opc) or is_call(opc) or is_loop(opc) or is_hlt(opc):
                    return True
                if getattr(instr, 'target_blocks', None):
                    return True
                return False
            bb.terminator = last_instr if last_instr and is_terminator_instr(last_instr) else None
            bb.successors = bb.successors or []
            bb.predecessors = bb.predecessors or []
            if not last_instr:
                # fallthrough to next bb if exists
                if idx + 1 < len(ordered_bb_ids):
                    bb.successors = [ordered_bbs[idx + 1]]
                continue
            opcode = (last_instr.opcode or '').upper()
            def resolve_operand_target(opd: Operand) -> Optional[str]:
                if getattr(opd, 'name', None) and opd.name in label_to_bb:
                    return label_to_bb[opd.name]
                if getattr(opd, 'memory', None) and isinstance(opd.memory.displacement, str):
                    disp = opd.memory.displacement
                    if disp in label_to_bb:
                        return label_to_bb[disp]
                return None
            targets: List[str] = []
            direct_target_found = False
            if is_unconditional(opcode) and last_instr.operands:
                t = resolve_operand_target(last_instr.operands[0])
                if t:
                    targets.append(t)
                    direct_target_found = True
            elif (is_conditional(opcode) or is_loop(opcode)) and last_instr.operands:
                t = resolve_operand_target(last_instr.operands[0])
                if t:
                    targets.append(t)
                    direct_target_found = True
                # conditional fallthrough to next block
                if idx + 1 < len(ordered_bb_ids):
                    targets.append(ordered_bb_ids[idx + 1])
            elif is_return(opcode) or is_hlt(opcode):
                targets = []
            elif is_call(opcode):
                # control returns to next block after call (fallthrough)
                if idx + 1 < len(ordered_bb_ids):
                    targets.append(ordered_bb_ids[idx + 1])
                # NOTE: call target annotation moved to separate pass below
            else:
                # default fallthrough
                if idx + 1 < len(ordered_bb_ids):
                    targets.append(ordered_bb_ids[idx + 1])
            # Annotate indirect terminators with metadata and deferred-analysis hints
            if last_instr and last_instr.operands:
                opc = opcode # already uppercased
                if is_unconditional(opc) or is_conditional(opc) or is_loop(opc) or is_call(opc):
                    target_op = last_instr.operands[0]
                    resolved = resolve_operand_target(target_op)
                    if resolved is None and getattr(target_op, 'name', None) is None:
                        # Truly indirect (register or memory target, not a direct symbol/name)
                        last_instr.indirect_jump_kind = "unknown" if is_call(opc) else "intraprocedural"
                        last_instr.indirect_targets = [] # Placeholder – later steps can populate via data-flow
            # Populate last_instr.target_blocks for branches (non-calls)
            if direct_target_found and not is_call(opcode):
                instr_targets = []
                seen_ids = set()
                for op in last_instr.operands:
                    t = resolve_operand_target(op)
                    if t and t not in seen_ids:
                        seen_ids.add(t)
                        instr_targets.append(t)
                if instr_targets:
                    last_instr.target_blocks = [bb_map[t] for t in instr_targets if t in bb_map]
            # Set bb.successors (deduped, preserving order)
            seen = set()
            succs: List[BasicBlock] = []
            for tid in targets:
                if tid and tid not in seen:
                    seen.add(tid)
                    if tid in bb_map:
                        succs.append(bb_map[tid])
            bb.successors = succs
        # Separate pass: annotate target_blocks on ALL CALL instructions (internal direct calls)
        for bb in func.basic_blocks:
            for instr in bb.instructions:
                opc = (instr.opcode or '').upper()
                if is_call(opc) and instr.operands:
                    instr_targets = []
                    seen_ids = set()
                    for op in instr.operands:
                        t_id = resolve_operand_target(op)
                        if t_id and t_id not in seen_ids:
                            seen_ids.add(t_id)
                            instr_targets.append(t_id)
                    if instr_targets:
                        instr.target_blocks = [bb_map[t_id] for t_id in instr_targets if t_id in bb_map]
    # Predecessors (invert successors)
    for bb in bb_map.values():
        bb.predecessors = bb.predecessors or []
    for func in func_map.values():
        for bb in func.basic_blocks:
            for succ in bb.successors:
                if bb.id not in [p.id for p in succ.predecessors]:
                    succ.predecessors.append(bb)
    # Unified symbol catalog (defined + external)
    symbol_catalog: Dict[str, Dict[str, Any]] = {}
    for name, info in data_symbol_map.items():
        symbol_catalog[name] = {'kind': 'data', 'def': info, 'visibility': 'global' if name in global_set else 'local'}
    for label_name, info in label_def_map.items():
        kind = 'function' if info.get('bb_id') and (label_name in global_set or any(f.entry_label == label_name for f in func_map.values())) else 'label'
        symbol_catalog[label_name] = {'kind': kind, 'def': info, 'visibility': 'global' if label_name in global_set else 'local'}
    for ext in sorted(extern_symbols):
        symbol_catalog[ext] = {'kind': 'external', 'visibility': 'global'}
    # Centralized operand symbol_ref annotation + external kind inference
    # We'll also create a label_map for quick mapping to Label objects (existing labels)
    label_map: Dict[str, Label] = {}
    for sec in ast_prog.sections:
        for child in sec.children:
            if isinstance(child, Function):
                for bb in child.basic_blocks:
                    if bb.start_label:
                        label_map[bb.start_label.name] = bb.start_label
    for instr in instr_map.values():
        opc = (instr.opcode or '').upper()
        is_call_instr = is_call(opc)
        for op in instr.operands:
            sym = None
            usage_kind = None
            if getattr(op, 'name', None) and op.name in symbol_catalog:
                sym = op.name
                # prefer mapping to an existing Label object if available
                if sym in label_map:
                    op.symbol_ref = label_map[sym]
                else:
                    # create a lightweight Label wrapper so serializer will emit symbol_ref
                    op.symbol_ref = Label(name=sym)
                if is_call_instr:
                    usage_kind = 'function'
            if getattr(op, 'memory', None) and isinstance(op.memory, Memory):
                disp = op.memory.displacement
                if isinstance(disp, str) and disp in symbol_catalog:
                    sym = disp
                    if sym in label_map:
                        op.symbol_ref = label_map[sym]
                    else:
                        op.symbol_ref = Label(name=sym)
                    # ensure via_got field exists (serializer will export it)
                    if not hasattr(op, 'via_got'):
                        op.via_got = False
                    usage_kind = 'data'
            if sym and symbol_catalog.get(sym, {}).get('kind') == 'external' and usage_kind:
                symbol_catalog[sym]['kind'] = usage_kind
    # Program-level symbol table (legacy dict-shaped)
    program_symbol_table: Dict[str, Any] = {}
    # Functions
    for func_id, func in func_map.items():
        entry = func.entry_label
        if entry:
            # find entry_bb if exists
            entry_bb_id = func.basic_blocks[0].id if func.basic_blocks else None
            program_symbol_table[entry] = {
                'kind': 'function',
                'definition': {'func_id': func_id, 'entry_bb': entry_bb_id},
                'visibility': 'global' if entry in global_set else 'local',
                'section': func.parent.name if func.parent else None,
                'location': func.location,
            }
    # Labels and other defs
    for name, info in label_def_map.items():
        rec = {'kind': info.get('kind', 'label'), 'visibility': 'global' if name in global_set else 'local'}
        if 'bb_id' in info:
            rec['definition'] = {'bb_id': info['bb_id'], 'section': info.get('section')}
        else:
            rec['definition'] = {'defined_in': info.get('defined_in'), 'section': info.get('section'), 'raw': info.get('definition')}
        if name not in program_symbol_table or program_symbol_table[name].get('kind') != 'function':
            program_symbol_table[name] = rec
        # else: preserve richer function metadata (func_id, entry_bb, etc.) set in the functions pass above
    # Data symbols
    for name, info in data_symbol_map.items():
        program_symbol_table.setdefault(name, {
            'kind': 'data',
            'definition': {'section': info.get('section'), 'raw': info.get('def')},
            'visibility': 'global' if name in global_set else 'local'
        })
    # externs
    for ext in sorted(extern_symbols):
        cat_kind = symbol_catalog.get(ext, {}).get('kind', 'external')
        program_symbol_table[ext] = {
            'kind': cat_kind,
            'visibility': 'global'
        }
    # Local label scoping (scoped_to anchors)
    for sec in ast_prog.sections:
        anchor = None
        for child in sec.children:
            if isinstance(child, Function):
                func = child
                entry = func.entry_label
                if entry:
                    anchor = entry
                for bb in func.basic_blocks:
                    sbl = bb.start_label.name if bb.start_label else None
                    if sbl:
                        if not sbl.startswith('.'):
                            anchor = sbl
                        else:
                            if sbl in program_symbol_table:
                                program_symbol_table[sbl].setdefault('scoped_to', anchor)
    # Attach symbol table and id_maps to AST program so ast_to_legacy_program_dict can include them
    ast_prog.symbol_table = program_symbol_table
    # id_maps in legacy shape
    id_maps = {
        'functions': {fid: {'section': f.parent.name if f.parent else None, 'entry_label': f.entry_label} for fid, f in func_map.items()},
        'basic_blocks': {bbid: {'function_id': next((fid for fid, f in func_map.items() if bbid in [b.id for b in f.basic_blocks]), None),
                                'start_label': bb_map[bbid].start_label.name if bb_map[bbid].start_label else None}
                         for bbid in bb_map},
    }
    ast_prog.id_maps = id_maps
    # Finally serialize back to legacy dict with enhancements included
    legacy_out = ast_to_legacy_program_dict(ast_prog, include_instr_locations=False, include_enhancements=True)
    return legacy_out
# ---- CLI entrypoint -----------------------------------------------------
def main():
    try:
        raw = sys.stdin.read()
        if not raw:
            eprint("No input on stdin. Expecting JSON legacy program dict.")
            sys.exit(2)
        obj = json.loads(raw)
    except json.JSONDecodeError as jde:
        eprint("Failed to parse JSON from stdin:", jde)
        sys.exit(3)
    try:
        enhanced = build_enhanced_program(obj)
        json.dump(enhanced, sys.stdout, indent=2, sort_keys=False)
        sys.stdout.write("\n")
    except Exception as exc:
        eprint("Unexpected error while enhancing AST:", repr(exc))
        raise
if __name__ == '__main__':
    main()
