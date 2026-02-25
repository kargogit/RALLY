# partitionAST.py
"""
Step 3: Partition typed AST (from astNodes.py) into Functions & BasicBlocks.

Usage:
    cat primitiveAST.json | python partitionAST.py > enhancedAST.json
"""

from typing import List, Dict, Any, Optional, Tuple, Set, Iterable
import itertools
import uuid
import dataclasses
import re

# Import dataclasses/types from astNodes.py (must be in same PYTHONPATH)
from astNodes import (
    Program,
    Section,
    LabelGroup,
    Label,
    Instruction,
    Immediate,
    BasicBlock,
    Function,
    ast_to_legacy_program_dict,
    legacy_program_dict_to_ast
)

_LABEL_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

# Config: which section names are considered executable (partitioned)
DEFAULT_EXECUTABLE_SECTIONS = {'.text'}


# -------------------------
# Helper functions
# -------------------------
def _extract_label_names_from_obj(obj, found: Set[str]) -> None:
    """
    Recursively inspect obj to find label-like strings and .name attributes.
    Adds candidates to `found`. Conservative: only accepts strings that match _LABEL_RE.
    Works for dicts, lists/tuples, dataclasses, and generic objects with attributes.
    """
    if obj is None:
        return

    # If it's already a known label-name string
    if isinstance(obj, str):
        if _LABEL_RE.match(obj):
            found.add(obj)
        return

    # If it's a dataclass, convert to dict to walk fields (safe)
    if dataclasses.is_dataclass(obj):
        try:
            od = dataclasses.asdict(obj)
        except Exception:
            # fallback to attribute iteration
            od = {k: getattr(obj, k) for k in dir(obj) if not k.startswith('_')}
        _extract_label_names_from_obj(od, found)
        return

    # If it's a dict: walk values
    if isinstance(obj, dict):
        for v in obj.values():
            _extract_label_names_from_obj(v, found)
        return

    # If it's a list/tuple/set: iterate elements
    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            _extract_label_names_from_obj(v, found)
        return

    # If it has a .name attribute that's a string, collect it
    name = getattr(obj, 'name', None)
    if isinstance(name, str) and _LABEL_RE.match(name):
        found.add(name)

    # If it has an 'operands' attribute (Instruction-like or data directive),
    # inspect it (covers cases like Instruction.operands or DataDirective.items).
    operands = getattr(obj, 'operands', None)
    if operands is not None:
        _extract_label_names_from_obj(operands, found)

    # Also try common names that may contain label refs (e.g., 'value', 'items', 'args')
    for attr in ('value', 'values', 'items', 'args', 'operands', 'displacement', 'label'):
        if hasattr(obj, attr):
            _extract_label_names_from_obj(getattr(obj, attr), found)

    # As a last resort, inspect simple public attributes (avoid callables, dunder)
    # but do this sparingly to avoid huge recursion/side effects.
    try:
        for k in dir(obj):
            if k.startswith('_'):
                continue
            # skip methods and descriptors
            v = getattr(obj, k)
            if callable(v):
                continue
            # small heuristic: only inspect simple attributes (str, dict, list, dataclass)
            if isinstance(v, (str, dict, list, tuple, set)) or dataclasses.is_dataclass(v):
                _extract_label_names_from_obj(v, found)
    except Exception:
        # swallow inspection errors (robust best-effort)
        pass


def _collect_label_references_from_sections(sections: Iterable[Section]) -> Set[str]:
    """
    Scan all sections' children for embedded label references (e.g. `dq constructor_stub`)
    and return a set of referenced label names.
    """
    found: Set[str] = set()
    for sec in sections:
        # Section has children which may be LabelGroup, Label, Instruction, or other nodes.
        for child in getattr(sec, 'children', []) or []:
            # If child is a LabelGroup, inspect its .instructions field (or equivalent)
            if isinstance(child, LabelGroup):
                for it in getattr(child, 'instructions', []) or []:
                    _extract_label_names_from_obj(it, found)
            else:
                # For any other child node (could be a 'function-node' etc), inspect generically
                _extract_label_names_from_obj(child, found)
        # Also scan pseudo_instruct for data directives (e.g. in .init_array)
        for item in getattr(sec, 'pseudo_instruct', []) or []:
            _extract_label_names_from_obj(item, found)
    return found





def _is_executable_section(section: Section) -> bool:
    return section.name in DEFAULT_EXECUTABLE_SECTIONS


def _opcode_family(opcode: str) -> str:
    """Return normalized opcode family (lowercase). Safe for None."""
    if not opcode:
        return ''
    return opcode.lower()


def _is_unconditional_jump(opcode: str) -> bool:
    """Detect unconditional jump opcodes. Conservative: startswith 'jmp'."""
    o = _opcode_family(opcode)
    return o.startswith('jmp')


def _is_return(opcode: str) -> bool:
    """Detect return opcodes."""
    o = _opcode_family(opcode)
    return o in ('ret', 'retq', 'retn', 'retl')


def _is_call(opcode: str) -> bool:
    o = _opcode_family(opcode)
    return o == 'call' or o.startswith('call')


def _is_loop(opcode: str) -> bool:
    """
    Detect loop-family instructions (loop, loope/loopz, loopne/loopnz).
    Conservative: treat any opcode that starts with 'loop' as a loop-branch.
    """
    o = _opcode_family(opcode)
    return o.startswith('loop')


def _is_branch(opcode: str) -> bool:
    """
    Consider conditional and unconditional jump-like opcodes:
    - conditional: 'je', 'jne', 'jg', etc. (start with 'j' but not 'jmp')
    - unconditional: 'jmp' handled separately
    We'll treat any opcode starting with 'j' as branch.
    """
    o = _opcode_family(opcode)
    return (o.startswith('j') and not _is_unconditional_jump(o)) or _is_unconditional_jump(o)


def _collect_operand_label_names(instr: Instruction) -> Set[str]:
    """Return set of operand label names referenced by an instruction."""
    names: Set[str] = set()
    for op in instr.operands:
        if getattr(op, 'name', None):
            names.add(op.name)
        # memory displacement or expression could embed names as strings;
        # best-effort: check .memory.displacement if it's a string
        mem = getattr(op, 'memory', None)
        if mem is not None:
            disp = getattr(mem, 'displacement', None)
            if isinstance(disp, str):
                names.add(disp)
    return names


import copy
import dataclasses

def _merge_source_locations(start_loc: Any, end_loc: Any) -> Any:
    """
    Create a new SourceLocation that spans from the start of `start_loc` to the
    end of `end_loc`. This implementation is robust to:
      - dict-style locations: { "start": {...}, "end": {...} }
      - dataclass-style locations (with 'start'/'end' or top-level end fields)
      - mixed combinations (dict vs dataclass)
    It prefers to return the same *type* as `start_loc` when possible and never
    mutates its inputs.
    """
    if not start_loc:
        return copy.deepcopy(end_loc)
    if not end_loc:
        return copy.deepcopy(start_loc)

    # Helper: try to extract a point (line/column-like) from an object
    def _point_from(obj):
        if obj is None:
            return None
        if isinstance(obj, dict):
            # either { "line":..., "column":... } or nested { "start":..., "end":... }
            if 'line' in obj or 'column' in obj:
                return {'line': obj.get('line'), 'column': obj.get('column')}
            if 'start' in obj and isinstance(obj['start'], dict):
                return {'line': obj['start'].get('line'), 'column': obj['start'].get('column')}
            # fallback: return the dict itself
            return copy.deepcopy(obj)
        # dataclass or object with attributes
        if dataclasses.is_dataclass(obj):
            # try common shapes: obj.end or obj.start
            if hasattr(obj, 'end') and getattr(obj, 'end') is not None:
                return _point_from(getattr(obj, 'end'))
            if hasattr(obj, 'line') or hasattr(obj, 'column'):
                return {'line': getattr(obj, 'line', None), 'column': getattr(obj, 'column', None)}
            if hasattr(obj, 'start'):
                return _point_from(getattr(obj, 'start'))
        # generic object with attributes
        if hasattr(obj, 'line') or hasattr(obj, 'column'):
            return {'line': getattr(obj, 'line', None), 'column': getattr(obj, 'column', None)}
        if hasattr(obj, 'end'):
            return _point_from(getattr(obj, 'end'))
        # last resort: return a deepcopy
        return copy.deepcopy(obj)

    # If start_loc is a dict, return a new dict with its 'end' replaced by end_loc's end
    if isinstance(start_loc, dict):
        merged = copy.deepcopy(start_loc)
        # If end_loc is dict with nested 'end', prefer that
        if isinstance(end_loc, dict):
            if 'end' in end_loc:
                merged['end'] = copy.deepcopy(end_loc['end'])
            elif 'line' in end_loc or 'column' in end_loc:
                merged['end'] = {'line': end_loc.get('line'), 'column': end_loc.get('column')}
            else:
                # fallback: attempt extracting a point
                merged['end'] = _point_from(end_loc)
        else:
            # end_loc is an object/dc: try to extract its end-point
            merged['end'] = _point_from(end_loc)
        return merged

    # If start_loc is a dataclass, try to use dataclasses.replace to set its 'end'
    if dataclasses.is_dataclass(start_loc):
        # prefer using an 'end' attribute if present
        end_candidate = None
        if isinstance(end_loc, dict) and 'end' in end_loc:
            end_candidate = end_loc['end']
        elif dataclasses.is_dataclass(end_loc) and hasattr(end_loc, 'end'):
            end_candidate = getattr(end_loc, 'end')
        else:
            end_candidate = _point_from(end_loc)

        try:
            if hasattr(start_loc, 'end'):
                # Replace the end field. dataclasses.replace will create a new instance.
                return dataclasses.replace(start_loc, end=end_candidate)
            # If there is no 'end' field, try a few alternative end-field names
            potential = {}
            for name in ('line_end', 'col_end', 'column_end', 'end_line', 'end_column', 'offset_end'):
                if hasattr(start_loc, name):
                    val = getattr(end_loc, name, None) if hasattr(end_loc, name) else None
                    if val is None:
                        # try extracting from nested
                        pt = _point_from(end_loc)
                        # choose a sensible scalar if available
                        if isinstance(pt, dict) and 'line' in pt and 'column' in pt:
                            # choose line/column mapping heuristically
                            if name in ('line_end', 'end_line'):
                                val = pt.get('line')
                            elif name in ('col_end', 'column_end', 'end_column'):
                                val = pt.get('column')
                    if val is not None:
                        potential[name] = val
            if potential:
                return dataclasses.replace(start_loc, **potential)
        except Exception:
            # best-effort: fall through to returning start_loc unchanged
            pass
        return start_loc

    # Fallback: start_loc is some other object (not dict, not dataclass)
    # Attempt to construct a small dict with start from start_loc and end from end_loc
    try:
        start_pt = _point_from(start_loc)
        end_pt = _point_from(end_loc)
        return {'start': start_pt, 'end': end_pt}
    except Exception:
        return copy.deepcopy(start_loc)


# -------------------------
# Core partitioning logic
# -------------------------
def _flatten_section_labelgroups(section: Section) -> List[Tuple[str, Any]]:
    """
    Flatten the ordered sequence of LabelGroup children in a section into
    a linear stream of items: tuples of ('label', Label) or ('instr', Instruction).
    """
    linear: List[Tuple[str, Any]] = []
    for child in section.children:
        if isinstance(child, LabelGroup):
            for it in child.instructions:
                if isinstance(it, Label):
                    linear.append(('label', it))
                elif isinstance(it, Instruction):
                    linear.append(('instr', it))
                else:
                    linear.append(('other', it))
        else:
            # Preserve existing Function nodes or other types
            linear.append(('function-node', child))
    return linear


def _identify_function_entry_labels(
    linear_stream: List[Tuple[str, Any]],
    program_globals: List[str],
    program_sections: Optional[Iterable[Section]] = None
) -> Dict[str, bool]:
    """
    Heuristic: function entry labels come from:
      - program globals (explicit)
      - call targets found in the linear_stream (call instructions)
      - label references embedded in data sections (.init_array/.fini_array etc)

    Returns a dict mapping label_name -> is_boundary.
    is_boundary is True if global, main/_start, or referenced in init/fini arrays.
    """
    globals_set = set(program_globals or [])
    call_targets: Set[str] = set()

    for kind, node in linear_stream:
        if kind == 'instr' and isinstance(node, Instruction):
            if _is_call(node.opcode):
                call_targets.update(_collect_operand_label_names(node))

    label_names_in_stream: Set[str] = {
        node.name for kind, node in linear_stream if kind == 'label' and isinstance(node, Label)
    }

    data_references: Set[str] = set()
    init_fini_references: Set[str] = set()

    if program_sections is not None:
        try:
            data_references = _collect_label_references_from_sections(program_sections)
            # Specifically scan init/fini/preinit arrays for boundary classification
            init_fini_sections = {'.init_array', '.fini_array', '.preinit_array'}
            for sec in program_sections:
                if sec.name in init_fini_sections:
                    for child in getattr(sec, 'children', []) or []:
                        _extract_label_names_from_obj(child, init_fini_references)
                    for item in getattr(sec, 'pseudo_instruct', []) or []:
                        _extract_label_names_from_obj(item, init_fini_references)
        except Exception:
            data_references = set()
            init_fini_references = set()

    candidates = (globals_set | call_targets | data_references) & label_names_in_stream

    # Determine boundary status
    boundary_candidates = (globals_set | init_fini_references | {'main', '_start'}) & candidates

    return {name: (name in boundary_candidates) for name in candidates}


def _split_linear_stream_into_function_segments(
    linear_stream: List[Tuple[str, Any]],
    entry_label_candidates: Set[str]
) -> List[List[Tuple[str, Any]]]:
    """
    Split the flattened linear stream into function segments based on entry label candidates.
    """
    segments: List[List[Tuple[str, Any]]] = []
    cur: List[Tuple[str, Any]] = []

    def flush_current():
        nonlocal cur
        if cur:
            segments.append(cur)
            cur = []

    for idx, (kind, node) in enumerate(linear_stream):
        if kind == 'function-node':
            flush_current()
            segments.append([(kind, node)])
            continue

        if kind == 'label' and isinstance(node, Label) and node.name in entry_label_candidates:
            flush_current()
            cur = [(kind, node)]
        else:
            if not cur:
                cur.append((kind, node))
            else:
                cur.append((kind, node))

    flush_current()
    return segments


def _partition_segment_into_function(segment: List[Tuple[str, Any]], program_globals: List[str], bb_id_counter: itertools.count, entry_boundary_map: Dict[str, bool]) -> Optional[Function]:
    """
    Partition a function segment into BasicBlocks and return a Function.
    Correctly calculates source location ranges for blocks and the function.
    """
    if not segment:
        return None

    if len(segment) == 1 and segment[0][0] == 'function-node':
        node = segment[0][1]
        return node if isinstance(node, Function) else None

    instrs: List[Instruction] = []
    label_to_inst_index: Dict[str, int] = {}
    label_name_to_labelobj: Dict[str, Label] = {}

    for item_kind, item in segment:
        if item_kind == 'label' and isinstance(item, Label):
            label_to_inst_index[item.name] = len(instrs)
            label_name_to_labelobj[item.name] = item
        elif item_kind == 'instr' and isinstance(item, Instruction):
            instrs.append(item)

    if not instrs:
        return None

    # Determine entry label
    entry_label: Optional[str] = None
    labels_at_zero = [name for name, idx in label_to_inst_index.items() if idx == 0]
    if labels_at_zero:
        entry_label = labels_at_zero[0]
    else:
        for name in program_globals or []:
            if name in label_to_inst_index:
                entry_label = name
                break

    # Leader-based partitioning
    leaders: Set[int] = {0}
    for name, idx in label_to_inst_index.items():
        if 0 <= idx < len(instrs):
            leaders.add(idx)

    for i, instr in enumerate(instrs):
        op = _opcode_family(instr.opcode)
        referenced_labels = _collect_operand_label_names(instr)
        for lbl in referenced_labels:
            tgt_idx = label_to_inst_index.get(lbl)
            if tgt_idx is not None:
                leaders.add(tgt_idx)

        if _is_return(op) or _is_unconditional_jump(op):
            continue

        if _is_branch(op) or _is_loop(op):
            next_idx = i + 1
            if next_idx < len(instrs):
                leaders.add(next_idx)

    sorted_leaders = sorted([l for l in leaders if 0 <= l < len(instrs)])
    basic_blocks: List[BasicBlock] = []

    labels_by_index: Dict[int, List[Label]] = {}
    for name, idx in label_to_inst_index.items():
        if 0 <= idx <= len(instrs):
            labels_by_index.setdefault(idx, []).append(label_name_to_labelobj[name])

    for li, start_idx in enumerate(sorted_leaders):
        end_idx = sorted_leaders[li + 1] if (li + 1) < len(sorted_leaders) else len(instrs)
        block_insts = instrs[start_idx:end_idx]

        if not block_insts:
            continue

        bb = BasicBlock(instructions=block_insts)
        bb.id = f"bb_{next(bb_id_counter)}"

        # Attach start label (prefer first one at this index)
        label_list = labels_by_index.get(start_idx)
        if label_list:
            bb.start_label = label_list[0]

        # --- Source Location Calculation for BasicBlock ---
        # Start at the Label (if present) or the first Instruction
        loc_start = None
        if label_list and getattr(label_list[0], 'location', None):
            loc_start = label_list[0].location
        elif block_insts and getattr(block_insts[0], 'location', None):
            loc_start = block_insts[0].location

        # End at the last Instruction
        loc_end = None
        if block_insts and getattr(block_insts[-1], 'location', None):
            loc_end = block_insts[-1].location

        if loc_start and loc_end:
            bb.location = _merge_source_locations(loc_start, loc_end)
        elif loc_start:
            bb.location = loc_start
        # --------------------------------------------------

        basic_blocks.append(bb)

    func = Function(basic_blocks=basic_blocks)
    if entry_label:
        func.entry_label = entry_label
        func.is_boundary = entry_boundary_map.get(entry_label, False)

    # --- Source Location Calculation for Function ---
    # Spans from first BasicBlock start to last BasicBlock end
    if basic_blocks:
        func_start = getattr(basic_blocks[0], 'location', None)
        func_end = getattr(basic_blocks[-1], 'location', None)

        if func_start and func_end:
            func.location = _merge_source_locations(func_start, func_end)
        elif func_start:
            func.location = func_start
    # ------------------------------------------------

    return func


def partition_program_into_functions_and_basic_blocks(program: Program) -> Program:
    """
    Main entry: mutate `program` in-place (and also return it) performing Step 3.
    """
    bb_id_counter = itertools.count()
    for sec in program.sections:
        if not _is_executable_section(sec):
            continue

        linear_stream = _flatten_section_labelgroups(sec)
        entry_boundary_map = _identify_function_entry_labels(linear_stream, program.globals or [], program.sections)
        entry_candidates = set(entry_boundary_map.keys())
        segments = _split_linear_stream_into_function_segments(linear_stream, entry_candidates)

        if not segments and linear_stream:
            segments = [linear_stream]

        new_children: List[Any] = []
        for seg in segments:
            func = _partition_segment_into_function(seg, program.globals or [], bb_id_counter, entry_boundary_map)
            if func is not None:
                new_children.append(func)
            else:
                # Fallback: preserve labels/instrs as LabelGroup if Function creation failed
                from astNodes import LabelGroup as LG
                lg = LG()
                for kind, node in seg:
                    if kind == 'label' and isinstance(node, Label):
                        lg.instructions.append(node)
                    elif kind == 'instr' and isinstance(node, Instruction):
                        lg.instructions.append(node)
                if lg.instructions:
                    new_children.append(lg)

        sec.children = new_children

    return program


# -------------------------
# Optional convenience API
# -------------------------
def partition_and_serialize(program: Program) -> Dict[str, Any]:
    partition_program_into_functions_and_basic_blocks(program)
    from astNodes import ast_to_legacy_program_dict
    return ast_to_legacy_program_dict(program)


if __name__ == '__main__':
    import sys
    import json
    import argparse

    def eprint(*args, **kwargs):
        print(*args, file=sys.stderr, **kwargs)

    parser = argparse.ArgumentParser(
        description="Partition typed AST into Functions and BasicBlocks and output legacy JSON."
    )
    # The user asked for flag name "-iloc". We support that and a more explicit long form.
    parser.add_argument(
        '-iloc',
        '--include-instr-locations',
        action='store_true',
        help='Include per-instruction source locations in the output JSON (verbose).'
    )
    args = parser.parse_args()

    try:
        raw = sys.stdin.read()
        if not raw:
            eprint("No input on stdin. Expecting JSON legacy program dict.")
            sys.exit(2)

        obj = json.loads(raw)
        program = legacy_program_dict_to_ast(obj)
        # Partition in-place
        partition_program_into_functions_and_basic_blocks(program)

        # Serialize with the requested verbosity
        out_dict = ast_to_legacy_program_dict(program, include_instr_locations=args.include_instr_locations)
        json.dump(out_dict, sys.stdout, indent=2)
        sys.stdout.write("\n")

    except json.JSONDecodeError as jde:
        eprint("Failed to parse JSON from stdin:", jde)
        sys.exit(3)
    except ValueError as ve:
        eprint("Input validation error:", ve)
        sys.exit(4)
    except Exception as exc:
        eprint("Unexpected error while partitioning AST:", repr(exc))
        sys.exit(1)
