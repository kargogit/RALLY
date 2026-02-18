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
    """All sections in the program (e.g., .text, .data, .bss, .init_array). The .text section (or first section if missing) is serialized first."""

    globals: List[str] = field(default_factory=list)
    """List of global symbol names exposed by the binary (e.g., ['main'])."""

    symbol_table: Dict[str, Any] = field(default_factory=dict)
    """Symbol metadata map (symbol name → details such as kind, visibility, definition). Only present/populated when include_enhancements=True."""

    id_maps: Dict[str, Any] = field(default_factory=dict)
    """Mapping of synthetic IDs to metadata for functions and basic blocks (e.g., 'functions' and 'basic_blocks' sub-maps). Only present when include_enhancements=True."""


@dataclass
class Section(ASTNode):
    name: str
    """Name of the section (e.g., '.text', '.data', '.bss')."""

    children: List[Union['LabelGroup', 'Function', 'DataGroup']] = field(default_factory=list)
    """Ordered list of top-level children: LabelGroup/Function for code, DataGroup for data sections."""

    pseudo_instruct: List[Dict[str, Any]] = field(default_factory=list)
    """List of pseudo-instructions/directives specific to the section (e.g., {'directive': 'default', 'params': ['rel']}, extern declarations, data definitions)."""

    location: Optional[Dict[str, Any]] = None
    """Optional source/location metadata (e.g., file offset or address range). Rarely used."""

    parent: Optional['Program'] = None
    """Back-reference to containing Program object (set during deserialization; not serialized)."""


@dataclass
class LabelGroup(ASTNode):
    """
    A contiguous block of labels and instructions (used for flat/non-CFG code).
    `instructions` is an ordered list containing Label or Instruction instances.
    """
    instructions: List[Union['Instruction', 'Label']] = field(default_factory=list)
    """Sequential mix of labels and instructions in the flat group."""

    parent: Optional['Section'] = None


@dataclass
class BasicBlock(ASTNode):
    """
    A basic block of instructions forming part of a control-flow graph.
    `instructions` is an ordered list containing Instruction instances.
    """
    instructions: List['Instruction'] = field(default_factory=list)
    """Ordered list of instructions in the block (non-branch instructions + optional terminator)."""

    id: Optional[str] = None
    """Synthetic unique identifier (e.g., 'bb_0'). Required/populated when include_enhancements=True; serialized as string."""

    start_label: Optional['Label'] = None
    """Label object that starts the block (if any). In typed AST: Label object; in legacy dict: string name under 'start_label'."""

    location: Optional[Dict[str, Any]] = None
    """Optional location/address metadata for the block."""

    parent: Optional['Function'] = None
    """Back-reference to containing Function (set during deserialization; serialized as parent function id when enhancements enabled)."""

    terminator: Optional['Instruction'] = None
    """The terminating/branch instruction of the block (if present). In typed AST: Instruction object; in legacy dict: string id."""

    successors: List['BasicBlock'] = field(default_factory=list)
    """List of successor BasicBlock objects. In typed AST: object references; in legacy dict: list of string ids."""

    predecessors: List['BasicBlock'] = field(default_factory=list)
    """List of predecessor BasicBlock objects. In typed AST: object references; in legacy dict: list of string ids."""


@dataclass
class ArgumentDescriptor(ASTNode):
    """One recovered argument."""
    kind: str  # 'register' | 'stack' | 'float' | 'unknown'
    location: Union[str, int]  # register name (e.g. 'rdi') or stack offset (e.g. 16)
    index: Optional[int] = None  # 0-based argument index
    inferred_type: Optional[str] = None  # e.g. 'i64', 'double', None
    first_use: Optional[Dict[str, Any]] = None
    # first_use example: {'bb': 'bb_0', 'instr': 'instr_12', 'location': {...}}


@dataclass
class StackSlot(ASTNode):
    """A local stack slot, spill, or padding region."""
    name: Optional[str] = None  # optional hint (e.g., from label usage)
    offset: int = 0  # negative offset from rbp (or positive from rsp if no rbp)
    size: int = 0
    alignment: int = 0
    kind: str = "local"  # 'local' | 'spill' | 'arg' | 'padding'
    register: Optional[str] = None  # e.g. 'RBX' for saved register spill
    index: int = 0  # e.g. callee-saved register index
    inferred_type: Optional[str] = None  # e.g., 'i32_signed', 'f64', 'ptr', 'unknown'; for the slot's refinement


@dataclass
class LiftedFunctionSignature(ASTNode):
    """Finalized LLVM function signature for lifting: ret_type (%State*). Set in Step 9."""
    return_type: str  # e.g., 'void', 'i64', 'float', 'double'
    attributes: List[str] = field(default_factory=list)  # e.g., ['noreturn', 'nounwind']


@dataclass
class Function(ASTNode):
    """
    A function that groups basic blocks into a control-flow graph.
    `basic_blocks` is an ordered list containing BasicBlock instances.
    """
    basic_blocks: List['BasicBlock'] = field(default_factory=list)
    """Ordered list of basic blocks belonging to the function."""

    entry_label: Optional[str] = None
    """Name of the entry label (e.g., 'main', 'print_help'). Stored as string."""

    location: Optional[Dict[str, Any]] = None
    """Optional location metadata for the function."""

    parent: Optional[Union['Section', 'Program']] = None
    """Back-reference to containing Section or Program (set during deserialization; serialized as parent name/id when enhancements enabled)."""

    id: Optional[str] = None
    """Synthetic unique identifier (e.g., 'func:main'). Populated when include_enhancements=True; serialized as string."""

    noreturn_kind: Optional[str] = field(default=None)
    """Indicates noreturn semantics if present (e.g., 'noreturn')."""

    arguments: List[ArgumentDescriptor] = field(default_factory=list)

    stack_slots: List[StackSlot] = field(default_factory=list)

    return_type: Optional[str] = None
    """Inferred return LLVM type, e.g. 'i64', 'void', 'double', or None if ambiguous."""

    uses_frame_pointer: bool = False
    """True if standard PUSH RBP / MOV RBP, RSP prolog detected."""

    abi_compliance: str = "unknown"
    """One of: 'standard', 'partial', 'custom', 'raw'."""

    lifted_signature: Optional[LiftedFunctionSignature] = None
    """Set in Step 9: finalized return type and LLVM attributes after interprocedural refinement."""


@dataclass
class Instruction(ASTNode):
    opcode: str
    """Instruction opcode (e.g., 'MOV', 'CALL', 'RET', 'LOCK INC')."""

    operands: List['Operand'] = field(default_factory=list)
    """List of operand objects (0–3 typically)."""

    prefix: Optional[str] = None
    """Optional instruction prefix (e.g., 'LOCK')."""

    location: Optional[Dict[str, Any]] = None
    """Optional per-instruction location metadata (controlled by include_instr_locations)."""

    parent: Optional['BasicBlock'] = None
    """Back-reference to containing BasicBlock (set during deserialization; serialized as parent bb id when enhancements enabled)."""

    target_blocks: List['BasicBlock'] = field(default_factory=list)
    """List of target BasicBlock objects for branch/call instructions. In typed AST: object references; in legacy dict: list of string ids."""

    indirect_targets: List['BasicBlock'] = field(default_factory=list)  # Placeholder for future possible targets

    indirect_jump_kind: Optional[str] = None  # 'intraprocedural' or 'unknown'

    id: Optional[str] = None
    """Synthetic unique identifier (e.g., 'instr_42'). Populated when include_enhancements=True; serialized as string."""

    op_refinement: Optional[str] = None  # e.g., 'i32_unsigned', 'f64'; for operation-specific width/signedness/float

    call_target_kind: Optional[str] = None  # 'internal' | 'external' | 'indirect' | 'unresolved'

    indirect_targets_over_approximated: bool = False


@dataclass
class Label(ASTNode):
    name: str
    """Label name (e.g., 'main', '.sum_loop')."""

    location: Optional[Dict[str, Any]] = None
    """Optional location metadata for the label."""

    parent: Optional[Union['LabelGroup', 'BasicBlock']] = None
    """Back-reference to containing LabelGroup or BasicBlock (set during deserialization; not serialized directly)."""

    id: Optional[str] = None
    """Synthetic identifier (defaults to 'label_{name}' via __post_init__). Serialized when enhancements enabled."""
    def __post_init__(self):
        if self.id is None:
            self.id = f"label_{self.name}"

@dataclass
class Operand(ASTNode):
    register: Optional[str] = None
    """Register name if operand is a register (e.g., 'RAX', 'RDI')."""

    memory: Optional['Memory'] = None
    """Memory addressing mode if operand is memory access."""

    integer: Optional['Immediate'] = None
    """Immediate integer value if operand is a constant."""

    expression: Optional[Any] = None
    """Complex expression (rare; may be str like 'numbers' or dict structure)."""

    size: Optional[str] = None
    """Explicit operand size override (e.g., 'BYTE', 'DWORD')."""

    name: Optional[str] = None
    """Symbolic name for call targets or similar (e.g., 'printf')."""

    symbol_ref: Optional['Label'] = None
    """Reference to a Label object for symbolic addresses. In typed AST: Label object; in legacy dict (enhancements): string name."""

    rip_relative: bool = False
    """Flag indicating RIP-relative addressing (serialized when enhancements enabled)."""

    via_got: bool = False
    """Flag indicating access via Global Offset Table (serialized explicitly when enhancements enabled)."""

    inferred_type: Optional[str] = None  # e.g., 'i8_signed', 'f32'; for the operand's value refinement (pointee for memory)

    address_refinement: Optional[str] = None  # e.g., 'ptr'; only for memory operands (address components)


@dataclass
class Memory(ASTNode):
    base: Optional[str] = None
    """Base register (e.g., 'RIP', 'RSI')."""

    index: Optional[str] = None
    """Index register for scaled indexing."""

    scale: Optional[int] = None
    """Scale factor for index (1, 2, 4, 8)."""

    displacement: Optional[Union[int, str]] = None
    """Displacement/offset: integer offset or symbol name (e.g., 'usage_msg')."""


@dataclass
class Immediate(ASTNode):
    value: Union[int, str]
    """Immediate value (int or hex string representation)."""

    type: str
    """Type classification (e.g., 'DECIMAL_INTEGER', 'HEX_INTEGER', 'BYTE')."""

    ascii: Optional[str] = None
    """Optional ASCII character representation for byte values (e.g., '-')."""


@dataclass
class Register(ASTNode):
    name: str
    """Register name (standalone register operand; rarely used directly)."""


@dataclass
class Name(ASTNode):
    value: str
    """Standalone symbolic name (rarely used directly)."""


@dataclass
class Expression(ASTNode):
    type: str
    """Expression type (e.g., 'additive')."""

    operands: List[ASTNode] = field(default_factory=list)
    """Sub-operands of the expression."""

    operator: Optional[str] = None
    """Operator if applicable."""


@dataclass
class GlobalDecl(ASTNode):
    name: str
    """Name of a global declaration (unused in current pipeline)."""

    location: Optional[Dict[str, Any]] = None
    """Optional location metadata."""


@dataclass
class DataDirective(ASTNode):
    """A single data directive (db/dw/dd/dq/resb/resw/resd/resq/align/times/global/extern/etc.)."""
    kind: str  # e.g., 'DB', 'DW', 'DD', 'DQ', 'RESB', 'RESW', 'RESD', 'RESQ', 'ALIGN', 'TIMES', 'GLOBAL', 'EXTERN'
    operands: List[Union[Immediate, str, Expression]] = field(default_factory=list)
    # - Immediate for numeric values
    # - str for string literals (e.g., "hello\n")
    # - Expression for symbolic expressions (rare in data, but supported)
    repeat: Optional[Immediate] = None  # For TIMES prefix; only Immediate (symbolic TIMES is rare and can be left unsupported initially)
    alignment: Optional[int] = None     # Only meaningful for ALIGN
    parent: Optional['DataGroup'] = None


@dataclass
class DataGroup(ASTNode):
    """Contiguous group of labels and data directives (used for .data/.rodata/.bss)."""
    items: List[Union['Label', 'DataDirective']] = field(default_factory=list)
    parent: Optional['Section'] = None


# ---------------------------------------------------------------------------
# Deterministic serializer: typed AST -> legacy dict shape (compact JSON)
# ---------------------------------------------------------------------------

def _serialize_argument_descriptor(ad: ArgumentDescriptor, include_enhancements: bool = False) -> Dict[str, Any]:
    res: Dict[str, Any] = {
        'kind': ad.kind,
        'location': ad.location,
    }
    if ad.index is not None:
        res['index'] = ad.index
    if ad.inferred_type is not None:
        res['inferred_type'] = ad.inferred_type
    if ad.first_use is not None:
        res['first_use'] = ad.first_use
    return res


def _serialize_stack_slot(ss: StackSlot, include_enhancements: bool = False) -> Dict[str, Any]:
    res: Dict[str, Any] = {
        'offset': ss.offset,
        'size': ss.size,
        'alignment': ss.alignment,
        'kind': ss.kind,
        'index': ss.index,
    }
    if ss.name is not None:
        res['name'] = ss.name
    if ss.register is not None:
        res['register'] = ss.register
    if include_enhancements:
        if ss.inferred_type is not None:
            res['inferred_type'] = ss.inferred_type
    return res


def _serialize_data_directive(dd: DataDirective, include_enhancements: bool = False) -> Dict[str, Any]:
    res: Dict[str, Any] = {'kind': dd.kind}
    if dd.operands:
        serialized_ops = []
        for op in dd.operands:
            if isinstance(op, Immediate):
                int_obj = {'value': op.value, 'type': op.type}
                if op.ascii is not None:
                    int_obj['ascii'] = op.ascii
                serialized_ops.append({'immediate': int_obj})
            elif isinstance(op, str):
                serialized_ops.append({'string': op})
            elif isinstance(op, Expression):
                # Fallback: serialize as raw dict if Expression is dict-like, else str
                serialized_ops.append({'expression': op.__dict__ if hasattr(op, '__dict__') else str(op)})
            else:
                serialized_ops.append(op)  # legacy passthrough
        res['operands'] = serialized_ops
    if dd.repeat is not None:
        int_obj = {'value': dd.repeat.value, 'type': dd.repeat.type}
        if dd.repeat.ascii is not None:
            int_obj['ascii'] = dd.repeat.ascii
        res['repeat'] = int_obj
    if dd.alignment is not None:
        res['alignment'] = dd.alignment
    if include_enhancements and dd.parent:
        # Optional: expose parent if needed
        pass
    return res

def _serialize_dgroup(dg: DataGroup, include_instr_locations: bool = False, include_enhancements: bool = False) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for it in dg.items:
        if isinstance(it, Label):
            item: Dict[str, Any] = {'label': it.name}
            if it.location:
                item['location'] = it.location
            if include_enhancements and it.id:
                item['id'] = it.id
            items.append(item)
        elif isinstance(it, DataDirective):
            items.append({'data_directive': _serialize_data_directive(it, include_enhancements=include_enhancements)})
    return {'dgroup': items}

def _serialize_operand(op: Operand, include_enhancements: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if op.register:
        out['register'] = op.register
    if op.size:
        out['size'] = op.size
    if op.expression is not None:
        out['expression'] = op.expression
    if op.integer is not None:
        int_obj = {'value': op.integer.value, 'type': op.integer.type}
        if getattr(op.integer, 'ascii', None) is not None:
            int_obj['ascii'] = op.integer.ascii
        out['integer'] = int_obj
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
    if include_enhancements:
        if op.symbol_ref:
            out['symbol_ref'] = op.symbol_ref.name
        out['rip_relative'] = op.rip_relative
        # expose via_got explicitly (bool)
        out['via_got'] = getattr(op, 'via_got', False)
        if op.value_refinement is not None:
            out['value_refinement'] = op.value_refinement  # serialized as 'pointee_refinement' equivalent for memory
        if op.address_refinement is not None:
            out['address_refinement'] = op.address_refinement
    return out


def _serialize_instruction(instr: Instruction, include_instr_locations: bool = False, include_enhancements: bool = False) -> Dict[str, Any]:
    res: Dict[str, Any] = {'opcode': instr.opcode}
    if instr.prefix:
        res['prefix'] = instr.prefix
    if instr.operands:
        res['operands'] = [_serialize_operand(o, include_enhancements=include_enhancements) for o in instr.operands]
    if include_enhancements:
        if instr.id is not None:
            res['id'] = instr.id
        res['target_blocks'] = [tb.id for tb in instr.target_blocks]
        if instr.indirect_targets:
            res['indirect_targets'] = [it.id for it in instr.indirect_targets]
        if instr.indirect_jump_kind is not None:
            res['indirect_jump_kind'] = instr.indirect_jump_kind
        if instr.parent:
            res['parent'] = instr.parent.id
        if instr.op_refinement is not None:
            res['op_refinement'] = instr.op_refinement
        if instr.call_target_kind is not None:
            res['call_target_kind'] = instr.call_target_kind
        if getattr(instr, 'indirect_targets_over_approximated', False):
            res['indirect_targets_over_approximated'] = True
    return res


def _serialize_basic_block(bb: BasicBlock, include_instr_locations: bool = False, include_enhancements: bool = False) -> Dict[str, Any]:
    """
    Serialize a BasicBlock. For each instruction we produce an item shaped like
      { "instruction": { ... }, "location": { ... }? }
    The presence of the per-instruction "location" entry is controlled by
    `include_instr_locations` to reduce verbosity when desired.
    """
    res: Dict[str, Any] = {}
    items: List[Dict[str, Any]] = []
    for instr in bb.instructions:
        instr_dict = _serialize_instruction(instr, include_instr_locations=include_instr_locations, include_enhancements=include_enhancements)
        item: Dict[str, Any] = {'instruction': instr_dict}
        if include_instr_locations and getattr(instr, 'location', None):
            item['location'] = instr.location
        items.append(item)
    res['instructions'] = items
    res['id'] = bb.id
    if bb.start_label:
        res['start_label'] = bb.start_label.name
    if bb.location:
        res['location'] = bb.location
    # Added: Serialize enhancements if requested
    res['id'] = bb.id
    if include_enhancements:
        if bb.terminator:
            res['terminator'] = bb.terminator.id
        res['successors'] = [s.id for s in bb.successors]
        res['predecessors'] = [p.id for p in bb.predecessors]
        if bb.parent:
            res['parent'] = bb.parent.id
    return res


def _serialize_function(func: Function, include_instr_locations: bool = False, include_enhancements: bool = False) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    res['basic_blocks'] = [_serialize_basic_block(bb, include_instr_locations=include_instr_locations, include_enhancements=include_enhancements) for bb in func.basic_blocks]
    if func.entry_label:
        res['entry_label'] = func.entry_label
    if func.location:
        res['location'] = func.location
    if include_enhancements:
        if func.id is not None:
            res['id'] = func.id
        if func.parent:
            if isinstance(func.parent, Section):
                res['parent'] = func.parent.name
        if func.noreturn_kind is not None:
            res['noreturn_kind'] = func.noreturn_kind
        if func.arguments:  # only if non-empty
            res['arguments'] = [_serialize_argument_descriptor(a) for a in func.arguments]
        res['stack_slots'] = [_serialize_stack_slot(s) for s in func.stack_slots]
        if func.return_type is not None:
            res['return_type'] = func.return_type
        if func.uses_frame_pointer:  # only if True (default False → omitted)
            res['uses_frame_pointer'] = True
        if func.abi_compliance != "unknown":  # only if analyzed to something else
            res['abi_compliance'] = func.abi_compliance
        if func.lifted_signature is not None:
            res['lifted_signature'] = {
                'return_type': func.lifted_signature.return_type,
                'attributes': func.lifted_signature.attributes
            }
    return res


def _serialize_lgroup(lg: LabelGroup, include_instr_locations: bool = False, include_enhancements: bool = False) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for it in lg.instructions:
        if isinstance(it, Label):
            item: Dict[str, Any] = {'label': it.name}
            if it.location:
                item['location'] = it.location
            if include_enhancements:
                item['id'] = it.id
            items.append(item)
        elif isinstance(it, Instruction):
            instr_dict = _serialize_instruction(it, include_instr_locations=include_instr_locations, include_enhancements=include_enhancements)
            item: Dict[str, Any] = {'instruction': instr_dict}
            if include_instr_locations and it.location:
                item['location'] = it.location
            items.append(item)
    return {'lgroup': items}


def ast_to_legacy_section_dict(section: Section, include_instr_locations: bool = False, include_enhancements: bool = False) -> Dict[str, Any]:
    sec: Dict[str, Any] = {'name': section.name}
    children = []
    for child in section.children:
        if isinstance(child, LabelGroup):
            children.append(_serialize_lgroup(child, include_instr_locations=include_instr_locations, include_enhancements=include_enhancements))
        elif isinstance(child, Function):
            children.append({'function': _serialize_function(child, include_instr_locations=include_instr_locations, include_enhancements=include_enhancements)})
        elif isinstance(child, DataGroup):
            children.append(_serialize_dgroup(child, include_enhancements=include_enhancements))
    if children:
        sec['children'] = children
    if section.pseudo_instruct:
        sec['pseudo_instruct'] = section.pseudo_instruct
    if section.location:
        sec['location'] = section.location
    return sec


def ast_to_legacy_program_dict(program: Program, include_instr_locations: bool = False, include_enhancements: bool = False) -> Dict[str, Any]:
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
    out.append({'section': ast_to_legacy_section_dict(text_sec, include_instr_locations=include_instr_locations, include_enhancements=include_enhancements)})
    out.append({'globals': program.globals})
    for s in program.sections:
        if s is text_sec:
            continue
        out.append({'section': ast_to_legacy_section_dict(s, include_instr_locations=include_instr_locations, include_enhancements=include_enhancements)})

    # Build final dict with metadata at top level (no wrapper, no list entries)
    result: Dict[str, Any] = {'program': out}
    if include_enhancements:
        result['symbol_table'] = program.symbol_table
        result['id_maps'] = program.id_maps
    return result


# ---------------------------------------------------------------------------
# Deterministic deserializer: legacy dict shape -> typed AST
# ---------------------------------------------------------------------------

def _deserialize_argument_descriptor(ad_dict: Dict[str, Any]) -> ArgumentDescriptor:
    return ArgumentDescriptor(
        kind=ad_dict['kind'],
        location=ad_dict['location'],
        index=ad_dict.get('index'),
        inferred_type=ad_dict.get('inferred_type'),
        first_use=ad_dict.get('first_use'),
    )


def _deserialize_stack_slot(ss_dict: Dict[str, Any]) -> StackSlot:
    return StackSlot(
        name=ss_dict.get('name'),
        offset=ss_dict['offset'],
        size=ss_dict['size'],
        alignment=ss_dict['alignment'],
        kind=ss_dict['kind'],
        register=ss_dict.get('register'),
        index=ss_dict.get('index', 0),
        refinement=ss_dict.get('refinement') if 'refinement' in ss_dict else None,
    )


def _deserialize_data_directive(dd_dict: Dict[str, Any], include_enhancements: bool = False) -> DataDirective:
    kind = dd_dict['kind']
    operands: List[Union[Immediate, str, Expression]] = []
    for op_dict in dd_dict.get('operands', []):
        if 'immediate' in op_dict:
            int_dict = op_dict['immediate']
            operands.append(Immediate(value=int_dict['value'], type=int_dict['type'], ascii=int_dict.get('ascii')))
        elif 'string' in op_dict:
            operands.append(op_dict['string'])
        elif 'expression' in op_dict:
            # Reconstruct minimal Expression or keep as dict
            operands.append(Expression(type='unknown', operands=[]))  # placeholder
        else:
            operands.append(op_dict)  # legacy passthrough
    repeat = None
    if 'repeat' in dd_dict:
        int_dict = dd_dict['repeat']
        repeat = Immediate(value=int_dict['value'], type=int_dict['type'], ascii=int_dict.get('ascii'))
    alignment = dd_dict.get('alignment')
    return DataDirective(kind=kind, operands=operands, repeat=repeat, alignment=alignment)

def _deserialize_dgroup(dg_dict: Dict[str, Any], include_enhancements: bool = False) -> DataGroup:
    if 'dgroup' not in dg_dict:
        raise ValueError("Expected 'dgroup' key in data group dict")
    items = dg_dict['dgroup']
    parsed_items: List[Union[Label, DataDirective]] = []
    for it in items:
        if 'label' in it:
            lbl = Label(name=it['label'], location=it.get('location'))
            if include_enhancements and 'id' in it:
                lbl.id = it['id']
            parsed_items.append(lbl)
        elif 'data_directive' in it:
            parsed_items.append(_deserialize_data_directive(it['data_directive'], include_enhancements=include_enhancements))
    return DataGroup(items=parsed_items)

def _deserialize_operand(op_dict: Dict[str, Any], include_enhancements: bool = False) -> Operand:
    op = Operand()
    if 'register' in op_dict:
        op.register = op_dict['register']
    if 'size' in op_dict:
        op.size = op_dict['size']
    if 'expression' in op_dict:
        op.expression = op_dict['expression']
    if 'integer' in op_dict:
        int_dict = op_dict['integer']
        op.integer = Immediate(value=int_dict['value'],
                               type=int_dict['type'],
                               ascii=int_dict.get('ascii'))
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
    if include_enhancements:
        if 'symbol_ref' in op_dict:
            op._temp_symbol_ref = op_dict['symbol_ref']
        op.rip_relative = op_dict.get('rip_relative', False)
        op.via_got = op_dict.get('via_got', False)
        op.value_refinement = op_dict.get('value_refinement')
        op.address_refinement = op_dict.get('address_refinement')
    return op

def _deserialize_instruction(instr_dict: Dict[str, Any], include_enhancements: bool = False) -> Instruction:
    opcode = instr_dict['opcode']
    prefix = instr_dict.get('prefix')
    operands_dicts = instr_dict.get('operands', [])
    operands = [_deserialize_operand(o, include_enhancements=include_enhancements) for o in operands_dicts]
    instr = Instruction(opcode=opcode, operands=operands, prefix=prefix)
    if include_enhancements:
        instr.id = instr_dict.get('id')
        instr._temp_target_blocks = instr_dict.get('target_blocks', [])
        instr.indirect_jump_kind = instr_dict.get('indirect_jump_kind')
        instr._temp_indirect_targets = instr_dict.get('indirect_targets', [])
        instr.op_refinement = instr_dict.get('op_refinement')
        instr.call_target_kind = instr_dict.get('call_target_kind')
        instr.indirect_targets_over_approximated = instr_dict.get('indirect_targets_over_approximated', False)
    return instr


def _deserialize_basic_block(bb_dict: Dict[str, Any], include_enhancements: bool = False) -> BasicBlock:
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
            instr_dict = entry['instruction']
            instr = _deserialize_instruction(instr_dict, include_enhancements=include_enhancements)
            if 'location' in entry:
                instr.location = entry['location']
            instructions.append(instr)
        elif isinstance(entry, dict):
            instr = _deserialize_instruction(entry, include_enhancements=include_enhancements)
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
    if include_enhancements:
        bb._temp_successors = bb_dict.get('successors', [])
        bb._temp_predecessors = bb_dict.get('predecessors', [])
        bb._temp_terminator = bb_dict.get('terminator')
    return bb


def _deserialize_function(func_dict: Dict[str, Any], include_enhancements: bool = False) -> Function:
    basic_blocks_dicts = func_dict.get('basic_blocks', [])
    basic_blocks = [_deserialize_basic_block(bb, include_enhancements=include_enhancements) for bb in basic_blocks_dicts]
    func = Function(basic_blocks=basic_blocks)
    if 'entry_label' in func_dict:
        func.entry_label = func_dict['entry_label']
    if 'location' in func_dict:
        func.location = func_dict['location']
    if include_enhancements:
        func.id = func_dict.get('id')
        if 'noreturn_kind' in func_dict:
            func.noreturn_kind = func_dict['noreturn_kind']
        func.arguments = [
            _deserialize_argument_descriptor(ad)
            for ad in func_dict.get('arguments', [])
        ]
        func.stack_slots = [
            _deserialize_stack_slot(ss)
            for ss in func_dict.get('stack_slots', [])
        ]
        func.return_type = func_dict.get('return_type')  # None if missing
        func.uses_frame_pointer = func_dict.get('uses_frame_pointer', False)
        func.abi_compliance = func_dict.get('abi_compliance', "unknown")
        if 'lifted_signature' in func_dict:
            ls_dict = func_dict['lifted_signature']
            func.lifted_signature = LiftedFunctionSignature(
                return_type=ls_dict['return_type'],
                attributes=ls_dict.get('attributes', [])
            )
    return func


def _deserialize_lgroup(lg_dict: Dict[str, Any], include_enhancements: bool = False) -> LabelGroup:
    if 'lgroup' not in lg_dict:
        raise ValueError("Expected 'lgroup' key in label group dict")
    items = lg_dict['lgroup']
    instructions: List[Union[Instruction, Label]] = []
    for it in items:
        location = it.get('location')
        if 'label' in it:
            lbl = Label(name=it['label'], location=location)
            if include_enhancements and 'id' in it:
                lbl.id = it['id']
            instructions.append(lbl)
        elif 'instruction' in it:
            instr_dict = it['instruction']
            instr = _deserialize_instruction(instr_dict, include_enhancements=include_enhancements)
            instr.location = location
            instructions.append(instr)
        else:
            raise ValueError(f"Invalid item in lgroup: {it.keys()}")
    return LabelGroup(instructions=instructions)


def _deserialize_section(sec_dict: Dict[str, Any], include_enhancements: bool = False) -> Section:
    if 'name' not in sec_dict:
        raise ValueError("Section dict missing 'name'")
    name = sec_dict['name']
    location = sec_dict.get('location')
    children = []
    children_dicts = sec_dict.get('children', [])
    for child_dict in children_dicts:
        if 'lgroup' in child_dict:
            children.append(_deserialize_lgroup(child_dict, include_enhancements=include_enhancements))
        elif 'function' in child_dict:
            children.append(_deserialize_function(child_dict['function'], include_enhancements=include_enhancements))
        elif 'dgroup' in child_dict:
            children.append(_deserialize_dgroup(child_dict, include_enhancements=include_enhancements))
    pseudo_instruct = sec_dict.get('pseudo_instruct', [])
    return Section(name=name, children=children, pseudo_instruct=pseudo_instruct, location=location)


def legacy_program_dict_to_ast(program_dict: Dict[str, Any], include_enhancements: bool = False) -> Program:
    if not isinstance(program_dict, dict) or 'program' not in program_dict:
        raise ValueError("Input must be a dict with 'program' key")
    items = program_dict['program']
    if not isinstance(items, list) or len(items) < 2:
        raise ValueError("Legacy program must have at least section + globals entries")

    # Extract metadata (direct top-level keys preferred)
    symbol_table: Dict[str, Any] = program_dict.get('symbol_table', {})
    id_maps: Dict[str, Any] = program_dict.get('id_maps', {})

    # Backward compatibility: fall back to old top-level "enhancements" wrapper if direct keys missing
    if 'enhancements' in program_dict:
        enh = program_dict['enhancements']
        if 'symbol_table' in enh:
            symbol_table = enh.get('symbol_table', symbol_table)
        if 'id_maps' in enh:
            id_maps = enh.get('id_maps', id_maps)

    sections: List[Section] = []
    globals_list: Optional[List[str]] = None

    for i, item in enumerate(items):
        if i == 1:
            if not isinstance(item, dict) or 'globals' not in item:
                raise ValueError("Second entry must be {'globals': [...]}")
            globals_list = item['globals']
            if not isinstance(globals_list, list):
                raise ValueError("'globals' must be a list of strings")
        elif 'section' in item:
            sec_dict = item['section']
            section = _deserialize_section(sec_dict, include_enhancements=include_enhancements)
            sections.append(section)
        else:
            raise ValueError("Unexpected entry in program")

    if globals_list is None:
        raise ValueError("Globals entry not found")

    program = Program(
        sections=sections,
        globals=globals_list,
        symbol_table=symbol_table,
        id_maps=id_maps  # NEW
    )

    # Post-deserialization linking
    if include_enhancements:
        bb_map: Dict[str, BasicBlock] = {}
        label_map: Dict[str, Label] = {}
        instr_map: Dict[str, Instruction] = {}

        # Helper to resolve symbol_refs robustly
        def _resolve_symbol_ref(op: Operand, lbl_map: Dict[str, Label]):
            if hasattr(op, '_temp_symbol_ref'):
                sym_name = op._temp_symbol_ref
                target_label = lbl_map.get(sym_name)

                if target_label:
                    op.symbol_ref = target_label
                else:
                    # If the symbol is not found in the local label map (e.g., it is a data symbol
                    # or an external function), we create a synthetic Label object to preserve the
                    # symbol reference name. This ensures that symbol_ref is not dropped during
                    # serialization for valid symbols that aren't code labels.
                    op.symbol_ref = Label(name=sym_name)

                del op._temp_symbol_ref

        # Pass 1: Populate all maps and set parent pointers.
        # We must populate the maps fully before resolving links to handle forward references.
        for section in program.sections:
            section.parent = program
            for child in section.children:
                child.parent = section
                if isinstance(child, Function):
                    for bb in child.basic_blocks:
                        bb.parent = child
                        bb_map[bb.id] = bb
                        if bb.start_label:
                            label_map[bb.start_label.name] = bb.start_label
                        for instr in bb.instructions:
                            instr.parent = bb
                            instr_map[instr.id] = instr
                elif isinstance(child, LabelGroup):
                    for it in child.instructions:
                        if isinstance(it, Label):
                            it.parent = child
                            label_map[it.name] = it
                        elif isinstance(it, Instruction):
                            it.parent = child
                            instr_map[it.id] = it

        # Pass 2: Resolve links using the fully populated maps.
        for section in program.sections:
            for child in section.children:
                if isinstance(child, Function):
                    for bb in child.basic_blocks:
                        # Resolve successors
                        if hasattr(bb, '_temp_successors'):
                            bb.successors = [bb_map[s_id] for s_id in bb._temp_successors if s_id in bb_map]
                            del bb._temp_successors

                        # Resolve predecessors
                        if hasattr(bb, '_temp_predecessors'):
                            bb.predecessors = [bb_map[p_id] for p_id in bb._temp_predecessors if p_id in bb_map]
                            del bb._temp_predecessors

                        # Resolve terminator
                        if hasattr(bb, '_temp_terminator'):
                            bb.terminator = instr_map.get(bb._temp_terminator)
                            del bb._temp_terminator

                        # Resolve instruction links
                        for instr in bb.instructions:
                            # Resolve target_blocks
                            if hasattr(instr, '_temp_target_blocks'):
                                instr.target_blocks = [bb_map[tb_id] for tb_id in instr._temp_target_blocks if tb_id in bb_map]
                                del instr._temp_target_blocks
                            # Resolve indirect_targets (placeholder, usually empty initially)
                            if hasattr(instr, '_temp_indirect_targets'):
                                instr.indirect_targets = [bb_map[tid] for tid in instr._temp_indirect_targets if tid in bb_map]
                                del instr._temp_indirect_targets
                            # Resolve operand symbol_refs
                            for op in instr.operands:
                                _resolve_symbol_ref(op, label_map)

                elif isinstance(child, LabelGroup):
                    # Resolve symbol_refs in LabelGroup instructions
                    for it in child.instructions:
                        if isinstance(it, Instruction):
                            for op in it.operands:
                                _resolve_symbol_ref(op, label_map)

    return program


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
