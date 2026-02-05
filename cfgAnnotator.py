## cfgAnnotator.py
# Enhance dictionary-based AST with navigation, CFG links and symbol table.
# Reads a legacy-program dict JSON from stdin (Step 3 output), and writes an
# enhanced JSON to stdout. Enhancements are JSON-safe identifiers (no Python
# object references / cycles), suitable for downstream passes.

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
UNCONDITIONAL_JUMPS = {'JMP'}
CONDITIONAL_BRANCH_PREFIX = 'J'  # JE, JNE, JL, JG, JO, JZ, etc (but not JMP)
CALL_OPS = {'CALL'}
RETURN_OPS = {'RET'}
LOOP_OPS = {'LOOP'}

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

# ---- Main enhancement pass -----------------------------------------------
def build_enhanced_program(legacy: Dict[str, Any]) -> Dict[str, Any]:
    program = json.loads(json.dumps(legacy))  # deep copy
    fid_ctr = [0]
    bb_ctr = [0]
    instr_ctr = [0]
    func_map: Dict[str, Dict[str, Any]] = {}
    bb_map: Dict[str, Tuple[str, Dict[str, Any]]] = {}  # bb_id -> (func_id, bb dict)
    instr_map: Dict[str, Tuple[str, Dict[str, Any]]] = {}  # instr_id -> (bb_id, instr dict)
    label_def_map: Dict[str, Dict[str, Any]] = {}
    data_symbol_map: Dict[str, Dict[str, Any]] = {}
    extern_symbols: set[str] = set()
    globals_list = find_globals(program)
    global_set = set(globals_list or [])
    section_list = []

    for sec in iter_program_sections(program):
        section_list.append(sec)
        for pi in sec.get('pseudo_instruct', []):
            if isinstance(pi, dict):
                if 'name' in pi:
                    name = pi['name']
                    data_symbol_map[name] = {'kind': 'data', 'section': sec.get('name'), 'def': pi}
                elif pi.get('directive') == 'extern':
                    params = pi.get('params', [])
                    if isinstance(params, list):
                        extern_symbols.update(params)

    for sec in section_list:
        children = sec.get('children', [])
        new_children = []
        section_name = sec.get('name')
        for child in children:
            if 'function' in child:
                func = child['function']
                entry_label = func.get('entry_label')
                func_id = f"func:{entry_label}" if entry_label else unique_id("func_", fid_ctr)
                func['id'] = func_id
                func['parent'] = section_name
                func_map[func_id] = func
                bbs = func.get('basic_blocks', [])
                for bb in bbs:
                    if 'id' not in bb:
                        bb['id'] = unique_id("bb_", bb_ctr)
                    bb_id = bb['id']
                    if bb.get('start_label'):
                        label_def_map[bb['start_label']] = {
                            'kind': 'label', 'defined_in': 'code', 'bb_id': bb_id, 'section': section_name
                        }
                    bb_map[bb_id] = (func_id, bb)
                    bb['parent'] = func_id
                    ins_entries = bb.get('instructions', [])
                    for it in ins_entries:
                        instr_dict = (it['instruction'] if isinstance(it, dict) and 'instruction' in it else it)
                        if 'id' not in instr_dict:
                            instr_dict['id'] = unique_id("instr_", instr_ctr)
                        instr_id = instr_dict['id']
                        instr_map[instr_id] = (bb_id, instr_dict)
                        instr_dict['parent'] = bb_id
                        # annotate rip_relative and normalize GOT-style displacements
                        for op in instr_dict.get('operands', []):
                            if isinstance(op, dict) and 'memory' in op:
                                mem = op['memory']
                                if isinstance(mem, dict) and mem.get('base', '').upper() == 'RIP':
                                    op['rip_relative'] = True
                                    op['via_got'] = False
                                    disp = mem.get('displacement')
                                    if isinstance(disp, str):
                                        low = disp.lower()
                                        if 'wrt' in low and 'got' in low:
                                            sym = disp.split('wrt', 1)[0].strip()
                                            if sym:
                                                mem['displacement'] = sym
                                                op['via_got'] = True
                new_children.append(child)
            elif 'lgroup' in child:
                lg = child['lgroup']
                for it in lg:
                    if 'label' in it:
                        name = it['label']
                        label_def_map[name] = {
                            'kind': 'label', 'defined_in': 'label_group',
                            'definition': it, 'section': section_name
                        }
                new_children.append(child)
            else:
                new_children.append(child)
        sec['children'] = new_children

    # Handle .init_array / .fini_array references
    for sec in section_list:
        for pi in sec.get('pseudo_instruct', []):
            for v in pi.get('values', []):
                if isinstance(v, dict) and 'symbol' in v:
                    sym = v['symbol']
                    if sym in label_def_map:
                        label_def_map[sym].setdefault('referenced_by', []).append({
                            'section': sec.get('name'), 'pseudo': pi
                        })

    label_to_bb: Dict[str, str] = {
        lbl: info['bb_id'] for lbl, info in label_def_map.items() if 'bb_id' in info
    }

    # CFG construction (successors, terminators, target_blocks)
    for func_id, func in func_map.items():
        bbs = func.get('basic_blocks', [])
        ordered_bb_ids = [bb['id'] for bb in bbs]
        bb_index = {bb_id: idx for idx, bb_id in enumerate(ordered_bb_ids)}
        for idx, bb in enumerate(bbs):
            bb_id = bb['id']
            ins_entries = bb.get('instructions', [])
            last_instr = None
            if ins_entries:
                last_entry = ins_entries[-1]
                last_instr = last_entry['instruction'] if 'instruction' in last_entry else last_entry

            def is_terminator_instr(instr: Dict[str, Any]) -> bool:
                if not instr:
                    return False
                opc = instr.get('opcode', '').upper()
                if is_unconditional(opc) or is_conditional(opc) or is_return(opc) or is_call(opc) or is_loop(opc):
                    return True
                if instr.get('target_blocks'):
                    return True
                return False

            bb['terminator'] = last_instr['id'] if last_instr and is_terminator_instr(last_instr) else None
            bb['successors'] = []
            bb['predecessors'] = bb.get('predecessors', []) or []
            if not last_instr:
                if idx + 1 < len(ordered_bb_ids):
                    bb['successors'].append(ordered_bb_ids[idx + 1])
                continue

            opcode = last_instr.get('opcode', '').upper()

            def resolve_operand_target(opd: Dict[str, Any]) -> Optional[str]:
                if 'name' in opd and opd['name'] in label_to_bb:
                    return label_to_bb[opd['name']]
                if 'memory' in opd and isinstance(opd['memory'], dict):
                    disp = opd['memory'].get('displacement')
                    if isinstance(disp, str) and disp in label_to_bb:
                        return label_to_bb[disp]
                return None

            targets: List[str] = []
            direct_target_found = False
            if is_unconditional(opcode) and last_instr.get('operands'):
                t = resolve_operand_target(last_instr['operands'][0])
                if t:
                    targets.append(t)
                    direct_target_found = True
            elif (is_conditional(opcode) or is_loop(opcode)) and last_instr.get('operands'):
                t = resolve_operand_target(last_instr['operands'][0])
                if t:
                    targets.append(t)
                    direct_target_found = True
                if idx + 1 < len(ordered_bb_ids):
                    targets.append(ordered_bb_ids[idx + 1])
            elif is_return(opcode):
                targets = []
            elif is_call(opcode):
                if idx + 1 < len(ordered_bb_ids):
                    targets.append(ordered_bb_ids[idx + 1])
                if last_instr.get('operands'):
                    t = resolve_operand_target(last_instr['operands'][0])
                    if t:
                        last_instr['target_blocks'] = [t]
            else:
                if idx + 1 < len(ordered_bb_ids):
                    targets.append(ordered_bb_ids[idx + 1])

            if direct_target_found and not is_call(opcode):
                instr_targets = [resolve_operand_target(op) for op in last_instr.get('operands', []) if resolve_operand_target(op)]
                if instr_targets:
                    seen = set()
                    last_instr['target_blocks'] = [t for t in instr_targets if t not in seen and not seen.add(t)]

            seen = set()
            bb['successors'] = [t for t in targets if t and t not in seen and not seen.add(t)]

    # Predecessors (invert successors)
    for bbid, (_, bb) in bb_map.items():
        bb.setdefault('predecessors', [])
    for func_id, func in func_map.items():
        for bb in func.get('basic_blocks', []):
            for succ in bb.get('successors', []):
                if succ in bb_map:
                    _, succ_bb = bb_map[succ]
                    succ_bb.setdefault('predecessors', [])
                    if bb['id'] not in succ_bb['predecessors']:
                        succ_bb['predecessors'].append(bb['id'])

    # Unified symbol catalog (defined + external)
    symbol_catalog: Dict[str, Dict[str, Any]] = {}
    for name, info in data_symbol_map.items():
        symbol_catalog[name] = {'kind': 'data', 'def': info, 'visibility': 'global' if name in global_set else 'local'}
    for label_name, info in label_def_map.items():
        kind = 'function' if info.get('bb_id') and (label_name in global_set or any(f.get('entry_label') == label_name for f in func_map.values())) else 'label'
        symbol_catalog[label_name] = {'kind': kind, 'def': info, 'visibility': 'global' if label_name in global_set else 'local'}
    for ext in sorted(extern_symbols):
        symbol_catalog[ext] = {'kind': 'external', 'visibility': 'global'}

    # Centralized operand symbol_ref annotation + external kind inference
    for instr_id, (bbid, instr) in instr_map.items():
        opc = instr.get('opcode', '').upper()
        is_call_instr = is_call(opc)
        for op in instr.get('operands', []):
            sym = None
            usage_kind = None
            if 'name' in op and op['name'] in symbol_catalog:
                sym = op['name']
                op['symbol_ref'] = sym
                if is_call_instr:
                    usage_kind = 'function'
            if 'memory' in op and isinstance(op['memory'], dict):
                disp = op['memory'].get('displacement')
                if isinstance(disp, str) and disp in symbol_catalog:
                    sym = disp
                    op['symbol_ref'] = disp
                    if 'via_got' not in op:
                        op['via_got'] = False
                    usage_kind = 'data'
            if sym and symbol_catalog[sym]['kind'] == 'external' and usage_kind:
                symbol_catalog[sym]['kind'] = usage_kind

    # Program-level symbol table
    program_symbol_table: Dict[str, Any] = {}
    for func_id, func in func_map.items():
        entry = func.get('entry_label')
        if entry:
            program_symbol_table[entry] = {
                'kind': 'function',
                'definition': {'func_id': func_id, 'entry_bb': func.get('basic_blocks', [{}])[0].get('id')},
                'visibility': 'global' if entry in global_set else 'local',
                'section': func.get('parent'),
                'location': func.get('location'),
            }
    for name, info in label_def_map.items():
        rec = {'kind': info.get('kind', 'label'), 'visibility': 'global' if name in global_set else 'local'}
        if 'bb_id' in info:
            rec['definition'] = {'bb_id': info['bb_id'], 'section': info.get('section')}
        else:
            rec['definition'] = {'defined_in': info.get('defined_in'), 'section': info.get('section'), 'raw': info.get('definition')}
        program_symbol_table[name] = rec
    for name, info in data_symbol_map.items():
        program_symbol_table.setdefault(name, {
            'kind': 'data',
            'definition': {'section': info.get('section'), 'raw': info.get('def')},
            'visibility': 'global' if name in global_set else 'local'
        })
    for ext in sorted(extern_symbols):
        cat_kind = symbol_catalog.get(ext, {}).get('kind', 'external')
        program_symbol_table[ext] = {
            'kind': cat_kind,
            'visibility': 'global'
        }

    # Local label scoping
    for sec in section_list:
        anchor = None
        for child in sec.get('children', []):
            if 'lgroup' in child:
                for it in child['lgroup']:
                    if 'label' in it:
                        lbl = it['label']
                        if not lbl.startswith('.'):
                            anchor = lbl
                        else:
                            if lbl in program_symbol_table:
                                program_symbol_table[lbl].setdefault('scoped_to', anchor)
            elif 'function' in child:
                func = child['function']
                entry = func.get('entry_label')
                if entry:
                    anchor = entry
                for bb in func.get('basic_blocks', []):
                    sbl = bb.get('start_label')
                    if sbl:
                        if not sbl.startswith('.'):
                            anchor = sbl
                        else:
                            if sbl in program_symbol_table:
                                program_symbol_table[sbl].setdefault('scoped_to', anchor)

    program['symbol_table'] = program_symbol_table
    program['id_maps'] = {
        'functions': {fid: {'section': f.get('parent'), 'entry_label': f.get('entry_label')} for fid, f in func_map.items()},
        'basic_blocks': {bbid: {'function_id': fid, 'start_label': bb.get('start_label')} for bbid, (fid, bb) in bb_map.items()},
    }
    return program

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
