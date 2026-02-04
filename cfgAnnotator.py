#!/usr/bin/env python3
"""
Step 4: Enhance dictionary-based AST with navigation, CFG links and symbol table.

Reads a legacy-program dict JSON from stdin (Step 3 output), and writes an
enhanced JSON to stdout. Enhancements are JSON-safe identifiers (no Python
object references / cycles), suitable for downstream passes.
"""
import sys
import json
from typing import Dict, List, Tuple, Any, Optional, Iterable

# ---- Utility helpers -----------------------------------------------------

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def unique_id(prefix: str, counter: List[int]) -> str:
    counter[0] += 1
    return f"{prefix}{counter[0]}"

def iter_program_sections(program: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for item in program.get('program', []):
        if isinstance(item, dict) and 'section' in item:
            yield item['section']

def find_globals(program: Dict[str, Any]) -> List[str]:
    items = program.get('program', [])
    if len(items) >= 2 and isinstance(items[1], dict) and 'globals' in items[1]:
        return items[1]['globals']
    return []

# ---- Opcode classification (simple, conservative) ------------------------

# Recognize instruction *names* that behave as unconditional jumps, returns,
# conditional branches, calls, loops. This is conservative; unknown forms are
# treated as non-resolving (no target).
UNCONDITIONAL_JUMPS = {'JMP'}
CONDITIONAL_BRANCH_PREFIX = 'J'  # JE, JNE, JL, JG, JO, JZ, etc (but not JMP)
CALL_OPS = {'CALL'}
RETURN_OPS = {'RET'}
LOOP_OPS = {'LOOP'}

def is_unconditional(opcode: str) -> bool:
    return opcode.upper() in UNCONDITIONAL_JUMPS

def is_conditional(opcode: str) -> bool:
    up = opcode.upper()
    return up != '' and up.startswith(CONDITIONAL_BRANCH_PREFIX) and up not in UNCONDITIONAL_JUMPS

def is_call(opcode: str) -> bool:
    return opcode.upper() in CALL_OPS

def is_return(opcode: str) -> bool:
    return opcode.upper() in RETURN_OPS

def is_loop(opcode: str) -> bool:
    return opcode.upper() in LOOP_OPS

# ---- Main enhancement pass -----------------------------------------------

def build_enhanced_program(legacy: Dict[str, Any]) -> Dict[str, Any]:
    # Work on a shallow copy to avoid mutating the user's original dict if caller keeps it.
    program = json.loads(json.dumps(legacy))  # cheap deep-copy to simplify in-place updates

    # Helpers to assign stable ids
    fid_ctr = [0]
    bb_ctr = [0]
    instr_ctr = [0]

    # maps
    func_map: Dict[str, Dict[str, Any]] = {}          # func_id -> function dict
    bb_map: Dict[str, Tuple[str, Dict[str, Any]]] = {} # bb_id -> (func_id, bb dict)
    instr_map: Dict[str, Tuple[str, Dict[str, Any]]] = {} # instr_id -> (bb_id, instr dict)
    label_def_map: Dict[str, Dict[str, Any]] = {}     # label name -> definition dict (info)
    data_symbol_map: Dict[str, Dict[str, Any]] = {}   # data name -> info

    # find globals list for visibility
    globals_list = find_globals(program)
    global_set = set(globals_list or [])

    # First pass: scan sections, functions, basic blocks, generate ids, collect labels and data
    section_list = []
    for sec in iter_program_sections(program):
        section_list.append(sec)
        # collect data symbols from pseudo_instruct (common in .data)
        for pi in sec.get('pseudo_instruct', []):
            if isinstance(pi, dict) and 'name' in pi:
                name = pi['name']
                data_symbol_map[name] = {'kind': 'data', 'section': sec.get('name'), 'def': pi}
    # Now go through sections and children for functions & label groups
    for sec in section_list:
        children = sec.get('children', [])
        new_children = []
        for child in children:
            if 'function' in child:
                func = child['function']
                # assign function id (prefer entry_label)
                entry_label = func.get('entry_label')
                if entry_label:
                    func_id = f"func:{entry_label}"
                else:
                    func_id = unique_id("func_", fid_ctr)
                func['_id'] = func_id
                func['_section'] = sec.get('name')
                func_map[func_id] = func

                # process basic blocks
                bbs = func.get('basic_blocks', [])
                for bb in bbs:
                    if 'id' not in bb:
                        bb['id'] = unique_id("bb_", bb_ctr)
                    bb_id = bb['id']
                    # ensure start_label presence consistency
                    start_label_name = bb.get('start_label')
                    if start_label_name:
                        # map label -> bb
                        label_def_map[start_label_name] = {'kind': 'label', 'defined_in': 'code', 'bb_id': bb_id, 'section': sec.get('name')}
                    bb_map[bb_id] = (func_id, bb)
                    bb['_function_id'] = func_id
                    bb['_section'] = sec.get('name')

                    # iterate instructions - assign instruction ids
                    ins_entries = bb.get('instructions', [])
                    for it in ins_entries:
                        # two permitted forms: {"instruction": {...}, "location": ...} or directly instruction dict
                        instr_dict = None
                        if isinstance(it, dict) and 'instruction' in it:
                            instr_dict = it['instruction']
                        elif isinstance(it, dict) and 'opcode' in it:
                            instr_dict = it
                        else:
                            raise ValueError("Unexpected instruction entry in basic block: {}".format(repr(it)))
                        if 'id' not in instr_dict:
                            instr_dict['id'] = unique_id("instr_", instr_ctr)
                        instr_id = instr_dict['id']
                        instr_map[instr_id] = (bb_id, instr_dict)
                        instr_dict['_bb_id'] = bb_id
                        instr_dict['_func_id'] = func_id

                        # annotate operand-level rip_relative (if memory.base == 'RIP')
                        for op in instr_dict.get('operands', []):
                            if isinstance(op, dict) and 'memory' in op:
                                mem = op['memory']
                                if isinstance(mem, dict) and mem.get('base') and mem['base'].upper() == 'RIP':
                                    op['rip_relative'] = True

                new_children.append(child)
            elif 'lgroup' in child:
                # LabelGroup: collect labels defined here
                lg = child['lgroup']
                # lg is a list of items
                for it in lg:
                    if 'label' in it:
                        name = it['label']
                        # a label in a label group - not necessarily attached to a bb, but record its definition
                        label_def_map[name] = {'kind': 'label', 'defined_in': 'label_group', 'definition': it, 'section': sec.get('name')}
                new_children.append(child)
            else:
                # unknown child (preserve)
                new_children.append(child)
        sec['children'] = new_children

    # Scan .init_array / .fini_array and pseudo_instruct symbol values for possible function registration targets
    # e.g. pseudo_instruct: { "dx": "dq", "values": [ { "symbol": "constructor_stub" } ] }
    for sec in section_list:
        pis = sec.get('pseudo_instruct', [])
        for pi in pis:
            if isinstance(pi, dict):
                vals = pi.get('values', [])
                for v in vals:
                    if isinstance(v, dict) and 'symbol' in v:
                        sym = v['symbol']
                        # if symbol name matches a function entry (we mapped bb start_label to label_def_map),
                        # keep the mapping but mark as referenced by data (so functions found by data will be known)
                        if sym in label_def_map:
                            label_def_map[sym].setdefault('referenced_by', []).append({'section': sec.get('name'), 'pseudo': pi})

    # Now create a symbol-to-function map for labels that correspond to function entries
    label_to_bb: Dict[str, str] = {}
    for lbl, info in list(label_def_map.items()):
        if info.get('bb_id'):
            label_to_bb[lbl] = info['bb_id']

    # Second pass: determine terminators and direct targets; compute successors for BBs
    for func_id, func in func_map.items():
        bbs = func.get('basic_blocks', [])
        # build an ordered list of bb ids in function order
        ordered_bb_ids = [bb['id'] for bb in bbs]
        bb_index = {bb_id: idx for idx, bb_id in enumerate(ordered_bb_ids)}

        for idx, bb in enumerate(bbs):
            bb_id = bb['id']
            ins_entries = bb.get('instructions', [])
            # find last instruction dict (instruction object)
            last_instr = None
            last_instr_entry = None
            if ins_entries:
                last_entry = ins_entries[-1]
                if 'instruction' in last_entry:
                    last_instr = last_entry['instruction']
                    last_instr_entry = last_entry
                else:
                    # bare instruction
                    last_instr = last_entry
                    last_instr_entry = last_entry
            # clear/initialize enhancement fields
            bb['terminator'] = None
            bb['successors'] = []
            bb['predecessors'] = bb.get('predecessors', []) or []

            if last_instr is None:
                # empty block (no instructions) - fallthrough to next if any
                if idx + 1 < len(ordered_bb_ids):
                    bb['successors'].append(ordered_bb_ids[idx+1])
                continue

            opcode = last_instr.get('opcode', '').upper()
            instr_id = last_instr.get('id')
            # mark terminator for bb
            # For our JSON-safe model, store terminator as instruction id (or None)
            bb['terminator'] = instr_id

            targets: List[str] = []

            # Helper: resolve an operand that references a label (operand with 'name')
            def resolve_operand_target(opd: Dict[str, Any]) -> Optional[str]:
                # if operand refers to a symbol name, prefer code bb mapping, else data symbol mapping
                if 'name' in opd:
                    nm = opd['name']
                    if nm in label_to_bb:
                        return label_to_bb[nm]
                    # if name is a function label referenced via call etc but not start_label, might exist in label_def_map
                    if nm in label_def_map and 'bb_id' in label_def_map[nm]:
                        return label_def_map[nm]['bb_id']
                    return None
                # memory displacement might be a symbol name as string
                if 'memory' in opd and isinstance(opd['memory'], dict):
                    disp = opd['memory'].get('displacement')
                    if isinstance(disp, str):
                        # displacement could be a symbol name
                        if disp in label_to_bb:
                            return label_to_bb[disp]
                        if disp in data_symbol_map:
                            return None  # data reference, not a direct code target
                        # otherwise unknown string - treat as unresolved
                return None

            # Determine if last instr has direct label target(s)
            direct_target_found = False
            if is_unconditional(opcode):
                # unconditional jump: single target if operand resolves to label
                ops = last_instr.get('operands', [])
                if ops:
                    t = resolve_operand_target(ops[0])
                    if t:
                        targets.append(t)
                        direct_target_found = True
                # else no resolved target
            elif is_conditional(opcode) or is_loop(opcode):
                # conditional branches and loops: one explicit target (operand), plus fall-through next
                ops = last_instr.get('operands', [])
                if ops:
                    t = resolve_operand_target(ops[0])
                    if t:
                        targets.append(t)
                        direct_target_found = True
                # fall-through
                if idx + 1 < len(ordered_bb_ids):
                    targets.append(ordered_bb_ids[idx+1])
            elif is_return(opcode):
                # returns: no successors
                targets = []
            elif is_call(opcode):
                # calls do not force a new leader / fall-through exists
                if idx + 1 < len(ordered_bb_ids):
                    targets.append(ordered_bb_ids[idx+1])
                # Attempt to resolve call target to mark symbol_ref (but do not treat as control-flow target)
                # We'll set instruction.target_blocks only if call target resolves to function entry.
                ops = last_instr.get('operands', [])
                if ops:
                    t = resolve_operand_target(ops[0])
                    if t:
                        # treat as target_blocks on instruction for cross reference, but do NOT add as bb successor
                        last_instr['target_blocks'] = [t]
                        direct_target_found = True
            else:
                # default / fallthrough: next basic block
                if idx + 1 < len(ordered_bb_ids):
                    targets.append(ordered_bb_ids[idx+1])

            # set instruction-level target_blocks for direct branch instructions
            if direct_target_found and not is_call(opcode):
                # compute instr target_blocks: gather unique target ids found via branch operand
                # Find operand(s) that resolved; we used 'targets' which may include fallthrough; for instr targets we want only direct branch targets
                instr_targets: List[str] = []
                ops = last_instr.get('operands', [])
                if ops:
                    # attempt to resolve each operand to bb
                    for op in ops:
                        tt = resolve_operand_target(op)
                        if tt:
                            instr_targets.append(tt)
                if instr_targets:
                    # dedupe while preserving order
                    seen = set()
                    deduped = []
                    for t in instr_targets:
                        if t not in seen:
                            seen.add(t)
                            deduped.append(t)
                    last_instr['target_blocks'] = deduped

            # Now set BB successors = targets (which we built to include fallthrough where appropriate)
            # Deduplicate while preserving order
            seen = set()
            succs: List[str] = []
            for t in targets:
                if t and t not in seen:
                    seen.add(t)
                    succs.append(t)
            bb['successors'] = succs

    # Third pass: compute predecessors by inverting successors
    for bbid, (fid, bb) in bb_map.items():
        # ensure predecessors list exists
        bb.setdefault('predecessors', [])
    for fid, func in func_map.items():
        for bb in func.get('basic_blocks', []):
            bbid = bb['id']
            for succ in bb.get('successors', []):
                if succ in bb_map:
                    _, succ_bb = bb_map[succ]
                    succ_bb.setdefault('predecessors', [])
                    if bbid not in succ_bb['predecessors']:
                        succ_bb['predecessors'].append(bbid)

    # Fourth pass: annotate operands with symbol_ref (name -> symbol table entry) and add rip_relative flags where memory base is RIP
    # Build a combined symbol catalog: functions (labels that map to bb), labels, data symbols
    symbol_catalog: Dict[str, Dict[str, Any]] = {}
    # data
    for name, info in data_symbol_map.items():
        symbol_catalog[name] = {'kind': 'data', 'def': info, 'visibility': ('global' if name in global_set else 'local')}
    # code labels (labels that map to bb)
    for label_name, info in label_def_map.items():
        if 'bb_id' in info:
            symbol_catalog[label_name] = {'kind': 'function' if label_name in globals_list or label_name in [f.get('entry_label') for f in func_map.values()] else 'label', 'def': info, 'visibility': ('global' if label_name in global_set else 'local')}
        else:
            symbol_catalog[label_name] = {'kind': 'label', 'def': info, 'visibility': ('global' if label_name in global_set else 'local')}

    # annotate instructions operands
    for instr_id, (bbid, instr) in instr_map.items():
        for op in instr.get('operands', []):
            if 'name' in op:
                nm = op['name']
                if nm in symbol_catalog:
                    op['symbol_ref'] = nm
            if 'memory' in op and isinstance(op['memory'], dict):
                mem = op['memory']
                disp = mem.get('displacement')
                if isinstance(disp, str):
                    if disp in symbol_catalog:
                        op['symbol_ref'] = disp
                # set rip_relative already set earlier, but ensure it's present
                if mem.get('base') and str(mem.get('base')).upper() == 'RIP':
                    op['rip_relative'] = True

    # Fifth pass: construct program-level symbol table with scoping for local labels
    # Strategy: program symbol table maps name -> entry with kind, definition id(s), visibility, section, type hints
    program_symbol_table: Dict[str, Any] = {}

    # Functions: each func has an entry_label — treat as top-level symbol
    for func_id, func in func_map.items():
        entry = func.get('entry_label')
        if entry:
            program_symbol_table[entry] = {
                'kind': 'function',
                'definition': {'func_id': func_id, 'entry_bb': (func.get('basic_blocks', [])[0].get('id') if func.get('basic_blocks') else None)},
                'visibility': ('global' if entry in global_set else 'local'),
                'section': func.get('_section'),
                'location': func.get('location'),
            }

    # labels and data
    for name, info in label_def_map.items():
        rec = {'kind': info.get('kind', 'label'), 'visibility': ('global' if name in global_set else 'local')}
        if info.get('bb_id'):
            rec['definition'] = {'bb_id': info['bb_id'], 'section': info.get('section')}
        else:
            rec['definition'] = {'defined_in': info.get('defined_in'), 'section': info.get('section'), 'raw': info.get('definition')}
        program_symbol_table[name] = rec

    for name, info in data_symbol_map.items():
        program_symbol_table.setdefault(name, {
            'kind': 'data',
            'definition': {'section': info.get('section'), 'raw': info.get('def')},
            'visibility': ('global' if name in global_set else 'local')
        })

    # NASM local label scoping: local labels often start with '.' or '@', scope to the nearest preceding global/non-local label.
    # We'll scan sections top-to-bottom and propagate the last non-local label as scope anchor.
    for sec in section_list:
        anchor = None
        for child in sec.get('children', []):
            # label groups define labels; functions also have entry_label
            if 'lgroup' in child:
                for it in child['lgroup']:
                    if 'label' in it:
                        lbl = it['label']
                        if not lbl.startswith('.'):
                            # non-local label -> update anchor
                            anchor = lbl
                        else:
                            # local label; record scoping
                            if lbl in program_symbol_table:
                                program_symbol_table[lbl].setdefault('scoped_to', anchor)
            elif 'function' in child:
                func = child['function']
                entry = func.get('entry_label')
                if entry:
                    # non-local label, becomes anchor
                    anchor = entry
                # inside function basic_blocks there may be start_label names
                for bb in func.get('basic_blocks', []):
                    sbl = bb.get('start_label')
                    if sbl:
                        if not sbl.startswith('.'):
                            anchor = sbl
                        else:
                            if sbl in program_symbol_table:
                                program_symbol_table[sbl].setdefault('scoped_to', anchor)

    # Attach symbol table to program
    program.setdefault('enhancements', {})
    program['enhancements']['symbol_table'] = program_symbol_table

    # Also attach summary maps for convenience
    program['enhancements']['id_maps'] = {
        'functions': {fid: {'section': f.get('_section'), 'entry_label': f.get('entry_label')} for fid, f in func_map.items()},
        'basic_blocks': {bbid: {'function_id': fid, 'start_label': bb.get('start_label')} for bbid, (fid, bb) in bb_map.items()},
    }

    # Make parent_id links for navigation (string ids only)
    # Functions: parent_id -> section name (string)
    for fid, func in func_map.items():
        func['parent_id'] = func.get('_section')
    # basic blocks: parent_id -> func id
    for bbid, (fid, bb) in bb_map.items():
        bb['parent_id'] = fid
    # instructions: parent_id -> bb id (instr dict already has _bb_id)
    for instr_id, (bbid, instr) in instr_map.items():
        instr['parent_id'] = bbid

    # Place back enhanced children into program structure (they are already mutated in-place)
    # Add a top-level enhancements flag
    #program.setdefault('metadata', {})
    #program['metadata']['enhanced_by'] = None
    #program['metadata']['enhanced_at'] = None  # caller can fill timestamp

    return program

# ---- CLI entrypoint -----------------------------------------------------

def main():
    parser_descr = "Step 4 AST enhancer: read legacy AST JSON from stdin, write enhanced JSON to stdout"
    # parse no arguments intentionally small
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
        # print pretty JSON
        json.dump(enhanced, sys.stdout, indent=2, sort_keys=False)
        sys.stdout.write("\n")
    except Exception as exc:
        eprint("Unexpected error while enhancing AST:", repr(exc))
        raise

if __name__ == '__main__':
    main()
