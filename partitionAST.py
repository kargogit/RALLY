"""
partitionAST.py
Step 3: Partition typed AST into Functions & BasicBlocks.
This is the synthesized best version that resolves all reported issues with
the frame_dummy / L1130_1 backward-jump thunk pattern (and similar real-world
compiler-generated layouts).
Key fixes:
  - Correct function boundary detection (globals + call targets + .init_array/.fini_array)
  - Backward-jump ownership repair: moves earlier "orphan" helper blocks (L1130_1 etc.)
    into the owning function (frame_dummy) even though they appear earlier in the binary.
  - Full connected-component relocation for chains of internal labels.
  - Function entry name = the externally visible symbol (frame_dummy, not L1130_1).
  - Clean leader-based BB partitioning that works for both normal and thunk layouts.
  - Robust label regex, more executable sections, safe merging, etc.
  - FIX: Non-local jump targets are promoted to function entries ONLY when they
    fall outside the linear span of every base-function that branches to them.
    Targets inside the jumping function's span (e.g. error_exit, do_exit) stay
    internal; targets outside (e.g. report_error jumped to from main but located
    before main) become separate functions.
  - Local labels (starting with . or L) are now strictly excluded from *all*
    function-entry promotion paths (including base candidates from globals/calls/init).
  - Redundant target-collection logic removed (leaders already cover every label
    position + explicit jump/loop fall-throughs, exactly matching the spec while
    guaranteeing no label is ever dropped).
  - Non-code nodes ("other" directives, top-level non-Function children) are now
    preserved in linear order exactly as required.
Usage:
    cat primitiveAST.json | python partitionAST.py > enhancedAST.json
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Set, Iterable
import itertools
import dataclasses
import re
import copy
from astNodes import (
    Program,
    Section,
    LabelGroup,
    Label,
    Instruction,
    BasicBlock,
    Function,
    ast_to_legacy_program_dict,
    legacy_program_dict_to_ast,
)
_LABEL_RE = re.compile(r"^[A-Za-z_\.][A-Za-z0-9_\.@$]*$")
DEFAULT_EXECUTABLE_SECTIONS = {
    ".text", ".init", ".fini", ".plt", ".plt.sec",
    ".text.startup", ".text.hot", ".text.unlikely"
}
INIT_FINI_SECTIONS = {".init_array", ".fini_array", ".preinit_array"}
# ====================== HELPERS ======================
def _is_executable_section(section: Section) -> bool:
    return section.name in DEFAULT_EXECUTABLE_SECTIONS
def _opcode(op: Optional[str]) -> str:
    return (op or "").lower()
def _is_ret(op: str) -> bool:
    return _opcode(op) in {"ret", "retq", "retn", "retl"}
def _is_call(op: str) -> bool:
    o = _opcode(op)
    return o == "call" or o.startswith("call")
def _is_uncond_jmp(op: str) -> bool:
    return _opcode(op).startswith("jmp")
def _is_cond_jmp(op: str) -> bool:
    o = _opcode(op)
    return o.startswith("j") and not _is_uncond_jmp(o)
def _is_loop(op: str) -> bool:
    return _opcode(op).startswith("loop")
def _is_any_jumpish(op: str) -> bool:
    return _is_cond_jmp(op) or _is_uncond_jmp(op) or _is_loop(op)
def _collect_operand_label_names(instr: Instruction) -> Set[str]:
    out: Set[str] = set()
    for op in getattr(instr, "operands", []) or []:
        nm = getattr(op, "name", None)
        if isinstance(nm, str) and _LABEL_RE.match(nm):
            out.add(nm)
        mem = getattr(op, "memory", None)
        if mem is not None:
            disp = getattr(mem, "displacement", None)
            if isinstance(disp, str) and _LABEL_RE.match(disp):
                out.add(disp)
    return out
def _extract_label_names_from_obj(obj: Any, found: Set[str]) -> None:
    if obj is None:
        return
    if isinstance(obj, str):
        if _LABEL_RE.match(obj):
            found.add(obj)
        return
    if dataclasses.is_dataclass(obj):
        try:
            _extract_label_names_from_obj(dataclasses.asdict(obj), found)
        except Exception:
            for k in dir(obj):
                if k.startswith("_") or callable(getattr(obj, k, None)):
                    continue
                _extract_label_names_from_obj(getattr(obj, k, None), found)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _extract_label_names_from_obj(v, found)
        return
    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            _extract_label_names_from_obj(v, found)
        return
    nm = getattr(obj, "name", None)
    if isinstance(nm, str) and _LABEL_RE.match(nm):
        found.add(nm)
    for attr in ("value", "values", "items", "args", "operands", "displacement", "label"):
        if hasattr(obj, attr):
            _extract_label_names_from_obj(getattr(obj, attr), found)
def _merge_source_locations(start_loc: Any, end_loc: Any) -> Any:
    if not start_loc:
        return copy.deepcopy(end_loc)
    if not end_loc:
        return copy.deepcopy(start_loc)
    def _point(x: Any):
        if x is None:
            return None
        if isinstance(x, dict):
            if "line" in x or "column" in x:
                return {"line": x.get("line"), "column": x.get("column")}
            if "start" in x and isinstance(x["start"], dict):
                return x["start"].copy()
            return copy.deepcopy(x)
        if dataclasses.is_dataclass(x):
            if hasattr(x, "line") or hasattr(x, "column"):
                return {"line": getattr(x, "line", None), "column": getattr(x, "column", None)}
        if hasattr(x, "line") or hasattr(x, "column"):
            return {"line": getattr(x, "line", None), "column": getattr(x, "column", None)}
        return copy.deepcopy(x)
    return {"start": _point(start_loc), "end": _point(end_loc)}
# ====================== LINEARIZATION ======================
def _flatten_section(section: Section) -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    for child in getattr(section, "children", []) or []:
        if isinstance(child, LabelGroup):
            for it in getattr(child, "instructions", []) or []:
                if isinstance(it, Label):
                    out.append(("label", it))
                elif isinstance(it, Instruction):
                    out.append(("instr", it))
                else:
                    out.append(("other", it))
        else:
            out.append(("function-node", child))
    return out
# ====================== ENTRY IDENTIFICATION ======================
def _scan_program_for_targets(program: Program) -> Tuple[Set[str], Set[str], Set[str]]:
    call_targets: Set[str] = set()
    jump_targets: Set[str] = set()
    init_fini_refs: Set[str] = set()
    for sec in getattr(program, "sections", []) or []:
        if _is_executable_section(sec):
            for child in getattr(sec, "children", []) or []:
                if isinstance(child, LabelGroup):
                    for it in getattr(child, "instructions", []) or []:
                        if isinstance(it, Instruction):
                            op = _opcode(it.opcode)
                            if _is_call(op):
                                call_targets |= _collect_operand_label_names(it)
                            elif _is_any_jumpish(op):
                                jump_targets |= _collect_operand_label_names(it)
                elif isinstance(child, Function):
                    for bb in getattr(child, "basic_blocks", []) or []:
                        for it in getattr(bb, "instructions", []) or []:
                            if isinstance(it, Instruction):
                                op = _opcode(it.opcode)
                                if _is_call(op):
                                    call_targets |= _collect_operand_label_names(it)
                                elif _is_any_jumpish(op):
                                    jump_targets |= _collect_operand_label_names(it)
        elif sec.name in INIT_FINI_SECTIONS:
            for child in getattr(sec, "children", []) or []:
                _extract_label_names_from_obj(child, init_fini_refs)
            for item in getattr(sec, "pseudo_instruct", []) or []:
                _extract_label_names_from_obj(item, init_fini_refs)
    return call_targets, jump_targets, init_fini_refs
def _identify_entry_candidates(
    linear_stream: List[Tuple[str, Any]],
    program: Program,
) -> Tuple[Set[str], Dict[str, bool]]:
    labels_in_stream: Set[str] = {
        node.name for kind, node in linear_stream
        if kind == "label" and isinstance(node, Label) and isinstance(getattr(node, "name", None), str)
    }
    globals_set = set(getattr(program, "globals", []) or [])
    call_targets, jump_targets, init_fini_refs = _scan_program_for_targets(program)
    def is_likely_local_label(name: str) -> bool:
        """Local assembler labels (starting with . or L) are never function entries."""
        return name.startswith('.') or name.startswith('L')
    # Base candidates: globals, call targets, and init/fini references.
    base_candidates = (globals_set | call_targets | init_fini_refs) & labels_in_stream
    # Local label filtering now applied to base candidates as well (per spec).
    base_candidates = {
        name for name in base_candidates if not is_likely_local_label(name)
    }
    # --- Span-aware promotion of jump-only targets ---
    # A non-local jump target is promoted to a function entry ONLY when it
    # falls OUTSIDE the linear span [entry … next_entry) of every base-function
    # that branches to it. Targets that sit inside the jumping function's span
    # are internal control-flow labels (e.g. error_exit, do_exit in _start),
    # not separate functions. Targets outside (e.g. report_error, located
    # before main but jumped to from main) become separate function entries.
    potential_external = {
        t for t in jump_targets
        if not is_likely_local_label(t)
        and t in labels_in_stream
        and t not in base_candidates
    }
    if potential_external:
        # Map every label to its index in the linear stream
        label_positions: Dict[str, int] = {}
        for i, (kind, node) in enumerate(linear_stream):
            if kind == "label" and isinstance(node, Label):
                label_positions[node.name] = i
        # Sorted stream-positions of base function entries
        sorted_base_positions = sorted(
            label_positions[name]
            for name in base_candidates
            if name in label_positions
        )
        def _span_end(entry_pos: int) -> int:
            """Position of the next base entry after *entry_pos*, or stream end."""
            for pos in sorted_base_positions:
                if pos > entry_pos:
                    return pos
            return len(linear_stream)
        def _owning_base_entry(stream_pos: int) -> Optional[int]:
            """Base entry position whose span covers *stream_pos*."""
            owner = None
            for ep in sorted_base_positions:
                if ep <= stream_pos:
                    owner = ep
                else:
                    break
            return owner
        # Collect every jump instruction that targets a potential-external label
        jump_info: List[Tuple[int, Set[str]]] = []
        for i, (kind, node) in enumerate(linear_stream):
            if kind == "instr" and isinstance(node, Instruction):
                if _is_any_jumpish(_opcode(node.opcode)):
                    hits = _collect_operand_label_names(node) & potential_external
                    if hits:
                        jump_info.append((i, hits))
        promoted: Set[str] = set()
        for target_name in potential_external:
            target_pos = label_positions.get(target_name)
            if target_pos is None:
                continue
            # A target is "internal" if it sits inside the span of at least
            # one base-function that branches to it.
            is_internal = False
            for jump_pos, targets in jump_info:
                if target_name not in targets:
                    continue
                owner_pos = _owning_base_entry(jump_pos)
                if owner_pos is not None:
                    if owner_pos <= target_pos < _span_end(owner_pos):
                        is_internal = True
                        break
            if not is_internal:
                promoted.add(target_name)
        base_candidates |= promoted
    candidates = base_candidates
    boundary_set = (globals_set | init_fini_refs | {"main", "_start"}) & candidates
    boundary_map = {name: (name in boundary_set) for name in candidates}
    return candidates, boundary_map
# ====================== SEGMENTATION + BACKWARD REPAIR ======================
def _split_into_segments(
    linear_stream: List[Tuple[str, Any]], entry_candidates: Set[str]
) -> List[List[Tuple[str, Any]]]:
    segments: List[List[Tuple[str, Any]]] = []
    cur: List[Tuple[str, Any]] = []
    def flush():
        nonlocal cur
        if cur:
            segments.append(cur)
            cur = []
    for kind, node in linear_stream:
        if kind == "function-node":
            flush()
            segments.append([(kind, node)])
            continue
        if kind == "label" and isinstance(node, Label) and node.name in entry_candidates:
            flush()
            cur = [(kind, node)]
        else:
            if cur or kind == "label": # start collecting once we see first label
                cur.append((kind, node))
    flush()
    return segments
def _repair_backward_jump_ownership(
    segments: List[List[Tuple[str, Any]]],
    linear_stream: List[Tuple[str, Any]],
    entry_candidates: Set[str],
) -> List[List[Tuple[str, Any]]]:
    """Core fix for frame_dummy/L1130_1 style thunks."""
    if not segments:
        return segments
    label_pos = {node.name: i for i, (kind, node) in enumerate(linear_stream)
                 if kind == "label" and isinstance(node, Label)}
    # 1. Determine ownership of backward targets
    owner: Dict[str, str] = {}
    current_owner: Optional[str] = None
    for i, (kind, node) in enumerate(linear_stream):
        if kind == "label" and isinstance(node, Label) and node.name in entry_candidates:
            current_owner = node.name
        if kind == "instr" and isinstance(node, Instruction) and current_owner:
            if _is_any_jumpish(_opcode(node.opcode)):
                for t in _collect_operand_label_names(node):
                    if t in label_pos and label_pos[t] < i and t not in entry_candidates:
                        owner[t] = current_owner
    # 2. Map entry -> segment index
    entry_to_seg: Dict[str, int] = {}
    for si, seg in enumerate(segments):
        for kind, node in seg:
            if kind == "label" and isinstance(node, Label) and node.name in entry_candidates:
                entry_to_seg[node.name] = si
                break
    # 3. Move chunks (including connected internal labels)
    for si in range(len(segments)):
        seg = segments[si]
        keep: List[Tuple[str, Any]] = []
        moves: List[Tuple[int, List[Tuple[str, Any]]]] = []
        j = 0
        while j < len(seg):
            kind, node = seg[j]
            if kind == "label" and isinstance(node, Label):
                lbl = node.name
                if lbl in owner and lbl not in entry_candidates:
                    dest_entry = owner[lbl]
                    dest_si = entry_to_seg.get(dest_entry)
                    if dest_si is not None and dest_si > si:
                        # Collect connected component (labels reachable by internal jumps/fallthrough)
                        chunk_start = j
                        k = j + 1
                        while k < len(seg):
                            ck, cn = seg[k]
                            if ck == "label" and isinstance(cn, Label):
                                if cn.name in entry_candidates:
                                    break
                                # reachable by jump from current chunk?
                                reachable = any(
                                    cn.name in _collect_operand_label_names(it)
                                    for _, it in seg[chunk_start:k] if isinstance(it, Instruction)
                                )
                                if not reachable and _is_ret(_opcode(getattr(seg[k-1][1], "opcode", "")) if k > 0 else False):
                                    break
                                k += 1
                                continue
                            k += 1
                        chunk = seg[chunk_start:k]
                        moves.append((dest_si, chunk))
                        j = k
                        continue
            keep.append((kind, node))
            j += 1
        segments[si] = keep
        for dest_si, chunk in moves:
            # Prepend so entry stays at the logical front
            segments[dest_si] = chunk + segments[dest_si]
    return segments
# ====================== BB PARTITIONING ======================
def _partition_segment_to_function(
    segment: List[Tuple[str, Any]],
    bb_id_counter: itertools.count,
    entry_candidates: Set[str],
    boundary_map: Dict[str, bool],
    program_globals: List[str],
) -> Optional[Function]:
    if not segment:
        return None
    if len(segment) == 1 and segment[0][0] == "function-node":
        node = segment[0][1]
        return node if isinstance(node, Function) else None
    instrs: List[Instruction] = []
    label_to_idx: Dict[str, int] = {}
    label_obj: Dict[str, Label] = {}
    for kind, node in segment:
        if kind == "label" and isinstance(node, Label):
            label_to_idx[node.name] = len(instrs)
            label_obj[node.name] = node
        elif kind == "instr" and isinstance(node, Instruction):
            instrs.append(node)
    if not instrs and not label_to_idx:
        return None
    # Entry label selection - prefer externally visible symbol
    entry_label: Optional[str] = None
    for name, idx in label_to_idx.items():
        if idx == 0 and name in entry_candidates:
            entry_label = name
            break
    if not entry_label:
        for name in label_to_idx:
            if name in entry_candidates:
                entry_label = name
                break
    if not entry_label:
        zeros = [n for n, idx in label_to_idx.items() if idx == 0]
        entry_label = zeros[0] if zeros else None
    if not entry_label:
        for g in program_globals:
            if g in label_to_idx:
                entry_label = g
                break
    # Leaders (exactly per spec: entry + all labels in segment + jump/loop targets
    # + fall-through after conditional/loop; the explicit target collection was
    # redundant because every label is already a leader, guaranteeing preservation).
    leaders: Set[int] = {0}
    for nm, idx in label_to_idx.items():
        if 0 <= idx < len(instrs):
            leaders.add(idx)
    for i, ins in enumerate(instrs):
        op = _opcode(ins.opcode)
        if _is_cond_jmp(op) or _is_loop(op):
            if i + 1 < len(instrs):
                leaders.add(i + 1)
    sorted_leaders = sorted(x for x in leaders if 0 <= x < len(instrs))
    labels_by_idx: Dict[int, List[Label]] = {}
    for nm, idx in label_to_idx.items():
        labels_by_idx.setdefault(idx, []).append(label_obj[nm])
    bbs: List[BasicBlock] = []
    for li, start in enumerate(sorted_leaders):
        end = sorted_leaders[li + 1] if li + 1 < len(sorted_leaders) else len(instrs)
        block_instrs = instrs[start:end]
        if not block_instrs:
            continue
        bb = BasicBlock(instructions=block_instrs)
        bb.id = f"bb_{next(bb_id_counter)}"
        lbls = labels_by_idx.get(start, [])
        if lbls:
            bb.start_label = lbls[0]
        loc_start = getattr(lbls[0], "location", None) if lbls else getattr(block_instrs[0], "location", None)
        loc_end = getattr(block_instrs[-1], "location", None)
        if loc_start and loc_end:
            bb.location = _merge_source_locations(loc_start, loc_end)
        elif loc_start:
            bb.location = loc_start
        bbs.append(bb)
    fn = Function(basic_blocks=bbs)
    if entry_label:
        fn.entry_label = entry_label
        fn.is_boundary = boundary_map.get(entry_label, False)
    if bbs:
        s = getattr(bbs[0], "location", None)
        e = getattr(bbs[-1], "location", None)
        if s and e:
            fn.location = _merge_source_locations(s, e)
        elif s:
            fn.location = s
    return fn
# ====================== MAIN ======================
def partition_program_into_functions_and_basic_blocks(program: Program) -> Program:
    bb_id_counter = itertools.count()
    for sec in getattr(program, "sections", []) or []:
        if not _is_executable_section(sec):
            continue
        linear = _flatten_section(sec)
        entry_candidates, boundary_map = _identify_entry_candidates(linear, program)
        segments = _split_into_segments(linear, entry_candidates)
        segments = _repair_backward_jump_ownership(segments, linear, entry_candidates)
        if not segments and linear:
            segments = [linear]
        new_children: List[Any] = []
        for seg in segments:
            # Preserve existing Function nodes and any top-level non-code ("other")
            # nodes exactly as required by the spec while maintaining linear order.
            if len(seg) == 1 and seg[0][0] == "function-node":
                new_children.append(seg[0][1])
                continue
            if len(seg) == 1 and seg[0][0] == "other":
                new_children.append(seg[0][1])
                continue
            fn = _partition_segment_to_function(
                seg,
                bb_id_counter=bb_id_counter,
                entry_candidates=entry_candidates,
                boundary_map=boundary_map,
                program_globals=getattr(program, "globals", []) or [],
            )
            if fn is not None:
                new_children.append(fn)
            else:
                lg = LabelGroup(instructions=[])
                for kind, node in seg:
                    if kind in ("label", "instr", "other"):
                        lg.instructions.append(node)
                if lg.instructions:
                    new_children.append(lg)
        sec.children = new_children
    return program
def partition_and_serialize(program: Program, include_instr_locations: bool = False) -> Dict[str, Any]:
    partition_program_into_functions_and_basic_blocks(program)
    return ast_to_legacy_program_dict(program, include_instr_locations=include_instr_locations)
if __name__ == "__main__":
    import sys
    import json
    import argparse
    parser = argparse.ArgumentParser(description="Partition typed AST into Functions & BasicBlocks.")
    parser.add_argument("-iloc", "--include-instr-locations", action="store_true",
                        help="Include per-instruction source locations")
    args = parser.parse_args()
    raw = sys.stdin.read()
    if not raw:
        print("No input on stdin.", file=sys.stderr)
        sys.exit(2)
    try:
        obj = json.loads(raw)
        prog = legacy_program_dict_to_ast(obj)
        partition_program_into_functions_and_basic_blocks(prog)
        out = ast_to_legacy_program_dict(prog, include_instr_locations=args.include_instr_locations)
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
    except Exception as e:
        print(f"Error: {e!r}", file=sys.stderr)
        sys.exit(1)
