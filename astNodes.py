"""
astNodes.py

Typed AST dataclasses and deterministic serializer for pipeline Step 2.

This module is intentionally self-contained and dependency-free (stdlib only).
It exposes dataclasses that represent the typed AST and a backwards-compatible
serializer function `ast_to_legacy_program_dict` used by downstream steps.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Dataclasses: stable, minimal, JSON-serializable primitives used across steps
# ---------------------------------------------------------------------------

@dataclass
class ASTNode:
    """Base class for typed AST nodes (marker)."""
    pass


@dataclass
class Program(ASTNode):
    sections: List['Section'] = field(default_factory=list)
    globals: List[str] = field(default_factory=list)


@dataclass
class Section(ASTNode):
    name: str
    children: List[Union['LabelGroup', 'Function']] = field(default_factory=list)
    pseudo_instruct: List[Dict[str, Any]] = field(default_factory=list)
    location: Optional[Dict[str, Any]] = None


@dataclass
class LabelGroup(ASTNode):
    """
    A contiguous block of labels and instructions.
    `instructions` is an ordered list containing Label or Instruction instances.
    """
    instructions: List[Union['Instruction', 'Label']] = field(default_factory=list)


@dataclass
class BasicBlock(ASTNode):
    """
    A basic block of instructions.
    `instructions` is an ordered list containing Instruction instances.
    """
    instructions: List['Instruction'] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(hash(frozenset())))  # Unique identifier
    start_label: Optional['Label'] = None
    location: Optional[Dict[str, Any]] = None


@dataclass
class Function(ASTNode):
    """
    A function that groups basic blocks.
    `basic_blocks` is an ordered list containing BasicBlock instances.
    """
    basic_blocks: List['BasicBlock'] = field(default_factory=list)
    entry_label: Optional[str] = None
    location: Optional[Dict[str, Any]] = None


@dataclass
class Instruction(ASTNode):
    opcode: str
    operands: List['Operand'] = field(default_factory=list)
    prefix: Optional[str] = None
    location: Optional[Dict[str, Any]] = None


@dataclass
class Label(ASTNode):
    name: str
    location: Optional[Dict[str, Any]] = None


@dataclass
class Operand(ASTNode):
    register: Optional[str] = None
    memory: Optional['Memory'] = None
    integer: Optional['Immediate'] = None
    expression: Optional[Any] = None
    size: Optional[str] = None
    name: Optional[str] = None


@dataclass
class Memory(ASTNode):
    base: Optional[str] = None
    index: Optional[str] = None
    scale: Optional[int] = None
    displacement: Optional[Union[int, str]] = None


@dataclass
class Immediate(ASTNode):
    value: Union[int, str]
    type: str


@dataclass
class Register(ASTNode):
    name: str


@dataclass
class Name(ASTNode):
    value: str


@dataclass
class Expression(ASTNode):
    type: str
    operands: List[ASTNode] = field(default_factory=list)
    operator: Optional[str] = None


@dataclass
class GlobalDecl(ASTNode):
    name: str
    location: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Deterministic serializer: typed AST -> legacy dict shape (compact JSON)
# ---------------------------------------------------------------------------

def _serialize_operand(op: Operand) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if op.register:
        out['register'] = op.register
    if op.size:
        out['size'] = op.size
    if op.expression is not None:
        out['expression'] = op.expression
    if op.integer is not None:
        out['integer'] = {'value': op.integer.value, 'type': op.integer.type}
    if op.name is not None:
        out['name'] = op.name
    if op.memory is not None:
        mem: Dict[str, Any] = {}
        if op.memory.base is not None:
            mem['base'] = op.memory.base
        if op.memory.index is not None:
            mem['index'] = op.memory.index
        if op.memory.scale is not None:
            mem['scale'] = op.memory.scale
        if op.memory.displacement is not None:
            mem['displacement'] = op.memory.displacement
        out['memory'] = mem
    return out


def _serialize_instruction(instr: Instruction) -> Dict[str, Any]:
    res: Dict[str, Any] = {'opcode': instr.opcode}
    if instr.prefix:
        res['prefix'] = instr.prefix
    if instr.operands:
        res['operands'] = [_serialize_operand(o) for o in instr.operands]
    return res


def _serialize_basic_block(bb: BasicBlock, include_instr_locations: bool = False) -> Dict[str, Any]:
    """
    Serialize a BasicBlock. For each instruction we produce an item shaped like
      { "instruction": { ... }, "location": { ... }? }
    The presence of the per-instruction "location" entry is controlled by
    `include_instr_locations` to reduce verbosity when desired.
    """
    res: Dict[str, Any] = {}
    items: List[Dict[str, Any]] = []
    for instr in bb.instructions:
        instr_dict = _serialize_instruction(instr)
        item: Dict[str, Any] = {'instruction': instr_dict}
        # conditionally preserve per-instruction location
        if include_instr_locations and getattr(instr, 'location', None):
            item['location'] = instr.location
        items.append(item)

    res['instructions'] = items
    res['id'] = bb.id
    if bb.start_label:
        res['start_label'] = bb.start_label.name
    if bb.location:
        res['location'] = bb.location
    return res


def _serialize_function(func: Function, include_instr_locations: bool = False) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    res['basic_blocks'] = [_serialize_basic_block(bb, include_instr_locations=include_instr_locations) for bb in func.basic_blocks]
    if func.entry_label:
        res['entry_label'] = func.entry_label
    if func.location:
        res['location'] = func.location
    return res


def _serialize_lgroup(lg: LabelGroup, include_instr_locations: bool = False) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for it in lg.instructions:
        if isinstance(it, Label):
            item: Dict[str, Any] = {'label': it.name}
            # label-level location stays (labels are higher-level than per-instruction)
            if it.location:
                item['location'] = it.location
            items.append(item)
        elif isinstance(it, Instruction):
            instr_dict = _serialize_instruction(it)
            item: Dict[str, Any] = {'instruction': instr_dict}
            # only include instruction-level location if requested
            if include_instr_locations and it.location:
                item['location'] = it.location
            items.append(item)
    return {'lgroup': items}


def ast_to_legacy_section_dict(section: Section, include_instr_locations: bool = False) -> Dict[str, Any]:
    sec: Dict[str, Any] = {'name': section.name}
    children = []
    for child in section.children:
        if isinstance(child, LabelGroup):
            # _serialize_lgroup already returns {'lgroup': items}
            children.append(_serialize_lgroup(child, include_instr_locations=include_instr_locations))
        elif isinstance(child, Function):
            children.append({'function': _serialize_function(child, include_instr_locations=include_instr_locations)})
    if children:
        sec['children'] = children
    if section.pseudo_instruct:
        sec['pseudo_instruct'] = section.pseudo_instruct
    if section.location:
        sec['location'] = section.location
    return sec


def ast_to_legacy_program_dict(program: Program, include_instr_locations: bool = False) -> Dict[str, Any]:
    """
    Backwards-compatible legacy shape:
      {"program": [ {"section": ...}, {"globals":[...]} , {"section": ...}, ... ]}

    Passing include_instr_locations=True will preserve per-instruction location
    objects in the produced legacy dict; default is False to reduce verbosity.
    """
    out: List[Union[Dict[str, Any], Any]] = []

    # find .text or fallback
    text_sec = None
    for s in program.sections:
        if s.name == '.text':
            text_sec = s
            break
    if text_sec is None and program.sections:
        text_sec = program.sections[0]
    if text_sec is None:
        text_sec = Section(name='.text')

    out.append({'section': ast_to_legacy_section_dict(text_sec, include_instr_locations=include_instr_locations)})
    out.append({'globals': program.globals})

    for s in program.sections:
        if s is text_sec:
            continue
        out.append({'section': ast_to_legacy_section_dict(s, include_instr_locations=include_instr_locations)})

    return {'program': out}


# ---------------------------------------------------------------------------
# Deterministic deserializer: legacy dict shape -> typed AST
# ---------------------------------------------------------------------------

def _deserialize_operand(op_dict: Dict[str, Any]) -> Operand:
    op = Operand()
    if 'register' in op_dict:
        op.register = op_dict['register']
    if 'size' in op_dict:
        op.size = op_dict['size']
    if 'expression' in op_dict:
        # Legacy format stores whatever was in expression (typically a dict tree
        # representing the expression). We preserve it as-is since Operand.expression
        # is typed as Any.
        op.expression = op_dict['expression']
    if 'integer' in op_dict:
        int_dict = op_dict['integer']
        op.integer = Immediate(value=int_dict['value'], type=int_dict['type'])
    if 'name' in op_dict:
        op.name = op_dict['name']
    if 'memory' in op_dict:
        mem_dict = op_dict['memory']
        mem = Memory()
        if 'base' in mem_dict:
            mem.base = mem_dict['base']
        if 'index' in mem_dict:
            mem.index = mem_dict['index']
        if 'scale' in mem_dict:
            mem.scale = mem_dict['scale']
        if 'displacement' in mem_dict:
            mem.displacement = mem_dict['displacement']
        op.memory = mem
    return op

def _deserialize_instruction(instr_dict: Dict[str, Any]) -> Instruction:
    opcode = instr_dict['opcode']
    prefix = instr_dict.get('prefix')
    operands_dicts = instr_dict.get('operands', [])
    operands = [_deserialize_operand(o) for o in operands_dicts]
    return Instruction(opcode=opcode, operands=operands, prefix=prefix)


def _deserialize_basic_block(bb_dict: Dict[str, Any]) -> BasicBlock:
    """
    Accept both the new wrapped form:
        {"instruction": {...}, "location": {...}?}
    and the old bare-form (legacy):
        {...}  (an instruction dict)
    This preserves backward compatibility with any older data.
    """
    instructions_entries = bb_dict.get('instructions', [])
    instructions: List[Instruction] = []

    for entry in instructions_entries:
        if isinstance(entry, dict) and 'instruction' in entry:
            # new wrapped form
            instr_dict = entry['instruction']
            instr = _deserialize_instruction(instr_dict)
            # restore per-instruction location if present
            if 'location' in entry:
                instr.location = entry['location']
            instructions.append(instr)
        elif isinstance(entry, dict):
            # legacy plain instruction dict form (keep working with old data)
            instr = _deserialize_instruction(entry)
            instructions.append(instr)
        else:
            raise ValueError(f"Invalid instruction entry in basic block: {repr(entry)}")

    bb = BasicBlock(instructions=instructions)
    if 'id' in bb_dict:
        bb.id = bb_dict['id']
    if 'start_label' in bb_dict:
        bb.start_label = Label(name=bb_dict['start_label'])
    if 'location' in bb_dict:
        bb.location = bb_dict['location']
    return bb

def _deserialize_function(func_dict: Dict[str, Any]) -> Function:
    basic_blocks_dicts = func_dict.get('basic_blocks', [])
    basic_blocks = [_deserialize_basic_block(bb) for bb in basic_blocks_dicts]
    func = Function(basic_blocks=basic_blocks)
    if 'entry_label' in func_dict:
        func.entry_label = func_dict['entry_label']
    if 'location' in func_dict:
        func.location = func_dict['location']
    return func


def _deserialize_lgroup(lg_dict: Dict[str, Any]) -> LabelGroup:
    if 'lgroup' not in lg_dict:
        raise ValueError("Expected 'lgroup' key in label group dict")
    items = lg_dict['lgroup']
    instructions: List[Union[Instruction, Label]] = []
    for it in items:
        location = it.get('location')
        if 'label' in it:
            instructions.append(Label(name=it['label'], location=location))
        elif 'instruction' in it:
            instr_dict = it['instruction']
            instr = _deserialize_instruction(instr_dict)
            instr.location = location
            instructions.append(instr)
        else:
            raise ValueError(f"Invalid item in lgroup: {it.keys()}")
    return LabelGroup(instructions=instructions)


def _deserialize_section(sec_dict: Dict[str, Any]) -> Section:
    if 'name' not in sec_dict:
        raise ValueError("Section dict missing 'name'")
    name = sec_dict['name']
    location = sec_dict.get('location')
    children = []
    children_dicts = sec_dict.get('children', [])
    for child_dict in children_dicts:
        if 'lgroup' in child_dict:
            # pass the child dict (which has 'lgroup' key) to the lgroup deserializer
            children.append(_deserialize_lgroup(child_dict))
        elif 'function' in child_dict:
            children.append(_deserialize_function(child_dict['function']))
    pseudo_instruct = sec_dict.get('pseudo_instruct', [])
    return Section(name=name, children=children, pseudo_instruct=pseudo_instruct, location=location)


def legacy_program_dict_to_ast(program_dict: Dict[str, Any]) -> Program:
    """
    Reverse of ast_to_legacy_program_dict.
    Reconstructs a Program instance from the legacy dictionary format.
    Preserves the deterministic section ordering used by the serializer:
      - sections appear in the order they are serialized ('.text' or fallback first,
        followed by remaining sections in original order).
    Raises ValueError on invalid input.
    """
    if not isinstance(program_dict, dict) or 'program' not in program_dict:
        raise ValueError("Input must be a dict with 'program' key")

    items = program_dict['program']
    if not isinstance(items, list) or len(items) < 2:
        raise ValueError("Legacy program must have at least section + globals entries")

    sections: List[Section] = []
    globals_list: Optional[List[str]] = None

    for i, item in enumerate(items):
        if i == 1:
            # Second entry is always globals
            if not isinstance(item, dict) or 'globals' not in item:
                raise ValueError("Second entry must be {'globals': [...]}")
            globals_list = item['globals']
            if not isinstance(globals_list, list):
                raise ValueError("'globals' must be a list of strings")
        else:
            # All other entries are sections
            if not isinstance(item, dict) or 'section' not in item:
                raise ValueError("Expected section entry {'section': {...}}")
            sec_dict = item['section']
            section = _deserialize_section(sec_dict)
            sections.append(section)

    if globals_list is None:
        raise ValueError("Globals entry not found")

    return Program(sections=sections, globals=globals_list)


def _validate_roundtrip_ast(ast_root: Program) -> None:
    """
    Validate that ast_to_legacy_program_dict(ast_root) -> legacy dict and then
    legacy_program_dict_to_ast(legacy) round-trips back to an AST equal to ast_root.

    Raises AssertionError on mismatch and writes helpful debugging output to stderr.
    """
    # produce legacy dict from original AST
    legacy = ast_to_legacy_program_dict(ast_root)

    # reconstruct AST from legacy dict
    reconstructed = legacy_program_dict_to_ast(legacy)

    # dataclass equality is used (dataclasses provide recursive equality)
    if reconstructed == ast_root:
        return

    # If we reach here, there's a mismatch: provide useful debug info then raise
    try:
        import json, sys
        sys.stderr.write("AST round-trip validation FAILED.\n")
        sys.stderr.write("Original AST serialized (legacy dict):\n")
        json.dump(legacy, sys.stderr, indent=2)
        sys.stderr.write("\n\nReconstructed AST serialized (legacy dict):\n")
        json.dump(ast_to_legacy_program_dict(reconstructed), sys.stderr, indent=2)
        sys.stderr.write("\n\n")
    except Exception:
        # best-effort debug; ignore JSON errors and fall back to repr
        import sys
        sys.stderr.write("AST round-trip validation FAILED (unable to JSON-dump).\n")
        sys.stderr.write(f"original repr: {repr(ast_root)}\n")
        sys.stderr.write(f"reconstructed repr: {repr(reconstructed)}\n")

    raise AssertionError("AST did not survive legacy round-trip (see stderr for details).")
