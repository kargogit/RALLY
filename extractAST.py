"""
asm_transformer.py – Step 2 of the NASM -> LLVM-IR pipeline

This updated module builds a typed, hierarchical AST (dataclasses)
while keeping the final serialized dict shape compatible with the
existing pipeline. A single, deterministic serialization step
(`ast_to_legacy_program_dict`) converts the typed AST into the
compact JSON-serialisable form (omitting `None`/empty fields and
preserving the dataclass field declaration order).
"""
# =============================================================================
# Imports
# =============================================================================
from typing import Any, Dict, List, Optional, Union, Callable, Protocol
from dataclasses import dataclass, field, asdict, is_dataclass
from abc import ABC, abstractmethod
from collections import defaultdict
import json
import sys

# =============================================================================
# Visitor protocol – defines the contract for concrete AST visitors
# =============================================================================
class ASTVisitor(Protocol):
    def visit_program(self, node: Any, context: Any) -> Any: ...
    def visit_section(self, node: Any, context: Any) -> Any: ...
    def visit_lgroup(self, node: Any, context: Any) -> Any: ...
    def visit_instruction(self, node: Any, context: Any) -> Any: ...
    def visit_operand(self, node: Any, context: Any) -> Any: ...
    def visit_label(self, node: Any, context: Any) -> Any: ...
    def generic_visit(self, node: Any, context: Any) -> Any: ...

# =============================================================================
# Immutable AST node definitions
# =============================================================================
@dataclass
class ASTNode:
    pass

@dataclass
class Program(ASTNode):
    sections: List['Section'] = field(default_factory=list)
    globals: List[str] = field(default_factory=list)

@dataclass
class Section(ASTNode):
    name: str
    lgroups: List['LabelGroup'] = field(default_factory=list)
    # Pseudo-instructions are kept as compact dicts for now (they are
    # already compact JSON shapes and numerous specialised forms exist).
    pseudo_instruct: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class LabelGroup(ASTNode):
    # A contiguous block of labels and instructions
    instructions: List[Union['Instruction', 'Label']] = field(default_factory=list)

@dataclass
class Instruction(ASTNode):
    opcode: str
    operands: List['Operand'] = field(default_factory=list)
    prefix: Optional[str] = None

@dataclass
class Label(ASTNode):
    name: str

@dataclass
class Operand(ASTNode):
    register: Optional[str] = None
    memory: Optional['Memory'] = None
    integer: Optional['Immediate'] = None
    expression: Optional[Any] = None  # ExpressionVisitor still returns dicts/strings
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

# Lightweight dataclass used for directive-based globals
@dataclass
class GlobalDecl(ASTNode):
    name: str

# =============================================================================
# Helper utilities – parsing-tree navigation
# =============================================================================
class ParseTreeNavigator:
    @staticmethod
    def normalize_token(token: Any) -> str:
        if isinstance(token, str):
            return token
        if isinstance(token, (tuple, list)) and len(token) >= 2:
            if isinstance(token[0], str):
                return str(token[0])
            if isinstance(token[1], list) and token[1]:
                return str(token[1][0])
        if isinstance(token, list) and len(token) == 1:
            return ParseTreeNavigator.normalize_token(token[0])
        if isinstance(token, dict):
            for key in ("name", "register", "size", "opcode",
                        "dx", "terminator_opcode"):
                if key in token and token[key]:
                    return ParseTreeNavigator.normalize_token(token[key])
        return ""

# =============================================================================
# Generic parse-tree visitor – provides the double-dispatch mechanism
# =============================================================================
class ParseTreeVisitor(ABC):
    def __init__(self):
        self.navigator = ParseTreeNavigator()

    def visit(self, node: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        if not isinstance(node, dict):
            return None
        context = context or {}
        for key in node.keys():
            method_name = f"visit_{key}"
            if hasattr(self, method_name):
                return getattr(self, method_name)(node[key], context)
        return self.generic_visit(node, context)

    def generic_visit(self, node: Any, context: Dict[str, Any]) -> Any:
        return node

# =============================================================================
# RIP-relative detection
# =============================================================================
class RipRelativeDetector:
    def __init__(self, navigator: ParseTreeNavigator):
        self.navigator = navigator

    def detect(self, operand_data: Any) -> Optional[Memory]:
        if not operand_data:
            return None
        if self._has_rip_or_rel(operand_data):
            displacement = self._extract_rip_displacement(operand_data)
            return Memory(base="RIP", displacement=displacement)
        return None

    def _has_rip_or_rel(self, node: Any) -> bool:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ('rel', 'rip'):
                    return True
                if self._has_rip_or_rel(value):
                    return True
        elif isinstance(node, list):
            return any(self._has_rip_or_rel(item) for item in node)
        return False

    def _extract_rip_displacement(self, operand_data: Any) -> Optional[Union[int, str]]:
        if isinstance(operand_data, list):
            for item in operand_data:
                if isinstance(item, dict) and 'expression' in item:
                    expr = item['expression']
                    if isinstance(expr, list) and expr:
                        return self._extract_from_expression(expr[0])
        return None

    def _extract_from_expression(self, expr_data: Any) -> Optional[Union[int, str]]:
        if not isinstance(expr_data, dict):
            return None
        if 'additiveExpression' in expr_data:
            add_expr = expr_data['additiveExpression']
            if len(add_expr) > 1:
                for part in add_expr:
                    if isinstance(part, dict):
                        res = self._extract_from_expression(part)
                        if res:
                            return res
        if 'multiplicativeExpression' in expr_data:
            return self._extract_from_expression(expr_data['multiplicativeExpression'][0])
        if 'castExpression' in expr_data:
            return self._extract_from_expression(expr_data['castExpression'][0])
        if 'unaryExpression' in expr_data and len(expr_data['unaryExpression']) > 1:
            return self._extract_from_expression(expr_data['unaryExpression'][1])
        if 'name' in expr_data:
            return self.navigator.normalize_token(expr_data['name'])
        if 'integer' in expr_data:
            int_data = expr_data['integer'][0]
            if isinstance(int_data, (list, tuple)) and len(int_data) == 2:
                return int_data[0]
        return None

# =============================================================================
# Transformation registry
# =============================================================================
class TransformationRegistry:
    def __init__(self):
        self._handlers: Dict[str, Callable] = {}

    def register(self, node_type: str):
        def decorator(func: Callable):
            self._handlers[node_type] = func
            return func
        return decorator

    def transform(self, node_type: str, data: Any,
                  context: Optional[Dict[str, Any]] = None) -> Any:
        handler = self._handlers.get(node_type)
        if handler:
            return handler(data, context or {})
        return None

# =============================================================================
# Expression visitor – unchanged behaviour (returns simple dicts/strings)
# =============================================================================
class ExpressionVisitor:
    def __init__(self, navigator: ParseTreeNavigator):
        self.navigator = navigator

    def process(self, expr_container: Dict[str, Any]) -> Any:
        if 'expression' not in expr_container:
            return None
        expr_list = expr_container['expression']
        if not expr_list or not isinstance(expr_list[0], dict):
            return None
        actual_expr = expr_list[0]
        if 'castExpression' in actual_expr:
            return self._visit_cast(actual_expr['castExpression'])
        if 'additiveExpression' in actual_expr:
            return self._visit_additive(actual_expr['additiveExpression'])
        if 'multiplicativeExpression' in actual_expr:
            return self._visit_multiplicative(actual_expr['multiplicativeExpression'])
        return None

    def _visit_cast(self, cast_expr: List[Any]) -> Any:
        expr = cast_expr[0]
        if 'name' in expr:
            return self.navigator.normalize_token(expr['name'])
        if 'register' in expr:
            return {'register': self.navigator.normalize_token(expr['register']).upper()}
        if 'integer' in expr:
            val = expr['integer'][0]
            return {'integer': {'type': val[1], 'value': val[0]}}
        if 'unaryExpression' in expr:
            return self._visit_unary(expr['unaryExpression'])
        return None

    def _visit_unary(self, unary_expr: List[Any]) -> Any:
        if len(unary_expr) == 2:
            op = self.navigator.normalize_token(unary_expr[0].get('unaryOperator', ''))
            operand = self.process({'expression': [unary_expr[1]]})
            if isinstance(operand, dict) and 'integer' in operand:
                operand['integer']['value'] = int(f"{op}{operand['integer']['value']}")
                return operand
            return {'unary_op': op, 'unary_val': operand}
        return None

    def _visit_additive(self, add_expr: List[Any]) -> Any:
        operands = []
        for comp in add_expr:
            if isinstance(comp, dict):
                if 'multiplicativeExpression' in comp or 'castExpression' in comp:
                    res = self.process({'expression': [comp]})
                    if isinstance(res, list):
                        res = {'multiplicative': res}
                    operands.append(res)
        return {'additive': operands} if len(operands) > 1 else (operands[0] if operands else None)

    def _visit_multiplicative(self, mul_expr: List[Any]) -> Any:
        operands = []
        for comp in mul_expr:
            if isinstance(comp, dict) and 'castExpression' in comp:
                operands.append(self.process({'expression': [comp]}))
        return operands if len(operands) > 1 else (operands[0] if operands else None)

# =============================================================================
# Concrete visitor – builds typed AST instances
# =============================================================================
class AsmTransformer(ParseTreeVisitor):
    def __init__(self):
        super().__init__()
        self.registry = TransformationRegistry()
        self.rip_detector = RipRelativeDetector(self.navigator)
        self._setup_handlers()
        self.text_section = Section(name='.text')

    def _setup_handlers(self):
        @self.registry.register('integer')
        def handle_integer(data, ctx):
            if isinstance(data, list) and data:
                int_data = data[0]
                if isinstance(int_data, (list, tuple)) and len(int_data) == 2:
                    return Immediate(value=int_data[0], type=int_data[1])
            return None

        @self.registry.register('label')
        def handle_label(data, ctx):
            return self.visit_label(data, ctx)

        @self.registry.register('operand')
        def handle_operand(data, ctx):
            return self.visit_operand(data, ctx)

        @self.registry.register('instruction')
        def handle_instruction(data, ctx):
            return self.visit_instruction(data, ctx)

    # Program entry point builds a Program dataclass
    def visit_program(self, program_data: List[Any], context: Dict[str, Any]) -> Program:
        prog = Program(sections=[self.text_section])
        context['current_section'] = self.text_section

        for node in program_data:
            if not isinstance(node, dict):
                continue
            result = self.visit(node, context)
            if result is None:
                continue
            if isinstance(result, Section):
                # do not duplicate the .text section
                if result.name != '.text':
                    prog.sections.append(result)
                context['current_section'] = result
            elif isinstance(result, GlobalDecl):
                prog.globals.append(result.name)
            elif isinstance(result, LabelGroup):
                ctx_sec: Section = context['current_section']
                ctx_sec.lgroups.append(result)
            elif isinstance(result, dict) and 'pseudo_instruct' in result:
                # some pseudo processing still returns compact dicts
                ctx_sec: Section = context['current_section']
                ctx_sec.pseudo_instruct.append(result['pseudo_instruct'])

        return prog

    def visit_line(self, line_data: List[Any], context: Dict[str, Any]) -> Optional[Any]:
        if not line_data or not isinstance(line_data[0], dict):
            return None
        line_content = line_data[0]
        if 'directive' in line_content:
            return self._process_directive(line_content['directive'], context)
        if 'pseudoinstruction' in line_content:
            return self._process_pseudoinstruction(line_content['pseudoinstruction'], context)
        return None

    def visit_block(self, lgroup_data: List[Any], context: Dict[str, Any]) -> Optional[LabelGroup]:
        lg = LabelGroup()
        for item in lgroup_data:
            if not isinstance(item, dict):
                continue
            if 'label' in item:
                label_node = self.visit_label(item['label'], context)
                if label_node:
                    lg.instructions.append(label_node)
            elif 'non_terminator_line' in item or 'terminator_line' in item:
                line_node = item.get('non_terminator_line') or item.get('terminator_line')
                instr_data = (line_node[0].get('instruction') or line_node[0].get('terminator_instruction'))
                if instr_data:
                    instr_node = self.visit_instruction(instr_data, context)
                    if instr_node:
                        lg.instructions.append(instr_node)
        return lg if lg.instructions else None

    def visit_label(self, label_data: List[Any], context: Dict[str, Any]) -> Optional[Label]:
        if not label_data or not isinstance(label_data[0], dict):
            return None
        name = self.navigator.normalize_token(label_data[0].get('name'))
        return Label(name=name)

    def visit_instruction(self, instr_data: List[Any], context: Dict[str, Any]) -> Optional[Instruction]:
        instr = Instruction(opcode='')
        for piece in instr_data:
            if not isinstance(piece, dict):
                continue
            if 'lock_prefix' in piece:
                instr.prefix = 'LOCK'
            elif 'opcode' in piece or 'terminator_opcode' in piece:
                key = 'opcode' if 'opcode' in piece else 'terminator_opcode'
                instr.opcode = self.navigator.normalize_token(piece[key]).upper()
            elif 'operand' in piece:
                op = self.visit_operand(piece['operand'], context)
                if op:
                    instr.operands.append(op)
        return instr

    def visit_operand(self, operand_data: List[Any], context: Dict[str, Any]) -> Operand:
        # detect rip-relative first
        rip_memory = self.rip_detector.detect(operand_data)
        if rip_memory:
            op = Operand(memory=rip_memory)
            return op

        op = Operand()
        for item in operand_data:
            if not isinstance(item, dict):
                continue
            if 'register' in item:
                op.register = self.navigator.normalize_token(item['register']).upper()
            elif 'size' in item:
                op.size = self.navigator.normalize_token(item['size']).upper()
            elif 'expression' in item:
                op.expression = self._process_expression(item)
            elif 'integer' in item:
                imm_node = self.registry.transform('integer', item['integer'])
                if imm_node:
                    op.integer = imm_node
            elif 'name' in item:
                op.name = self.navigator.normalize_token(item['name'])
        return op

    def _process_directive(self, directive_data: List[Any], context: Dict[str, Any]) -> Optional[Union[Section, GlobalDecl]]:
        if not directive_data or len(directive_data) < 2:
            return None
        directive_type = directive_data[0]
        if 'section' in directive_type:
            name = self._extract_section_name(directive_data)
            if name == '.text':
                return self.text_section
            new_section = Section(name=name)
            return new_section
        if 'global' in directive_type:
            name = self._extract_global_name(directive_data)
            return GlobalDecl(name=name)
        return None

    def _process_pseudoinstruction(self, pseudo_data: List[Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pseudo_dict: Dict[str, Any] = {}
        pseudo_values: List[Any] = []
        equ_pending = False
        for item in pseudo_data:
            if not isinstance(item, dict):
                continue
            if 'name' in item:
                pseudo_dict['name'] = self.navigator.normalize_token(item['name'])
            elif 'dx' in item:
                pseudo_dict['dx'] = self.navigator.normalize_token(item['dx'])
            elif 'resx' in item:
                pseudo_dict['resx'] = self.navigator.normalize_token(item['resx'])
            elif 'equ' in item:
                equ_pending = True
            elif 'expression' in item and equ_pending:
                equ_expr = self._build_equ_expression(item['expression'])
                if equ_expr:
                    pseudo_dict['equ'] = {'expression': equ_expr}
                equ_pending = False
            elif 'value' in item:
                val = self._process_pseudo_value(item['value'])
                if val:
                    pseudo_values.append(val)
            elif 'integer' in item:
                imm = self.registry.transform('integer', item['integer'])
                if imm:
                    pseudo_dict['integer'] = {'type': imm.type, 'value': imm.value}
        if pseudo_values:
            pseudo_dict['values'] = pseudo_values
        return {'pseudo_instruct': pseudo_dict} if pseudo_dict else None

    def _build_equ_expression(self, expr_container: List[Any]) -> Optional[Dict[str, Any]]:
        if not expr_container or not isinstance(expr_container[0], dict):
            return None
        actual = expr_container[0]
        if 'additiveExpression' not in actual:
            return None
        add_expr = actual['additiveExpression']
        operands: List[Dict[str, str]] = []
        operators: List[str] = []
        for part in add_expr:
            if isinstance(part, dict):
                if 'castExpression' in part:
                    ce = part['castExpression']
                    if ce and isinstance(ce[0], (list, tuple)):
                        tok = ce[0][0]
                        if tok == '$':
                            operands.append({'symbol': '$'})
                        else:
                            operands.append({'symbol': str(tok)})
                    elif ce and isinstance(ce[0], dict):
                        inner = ce[0]
                        if 'name' in inner:
                            name = self.navigator.normalize_token(inner['name'])
                            operands.append({'symbol': name})
                        elif 'register' in inner:
                            operands.append({'symbol': self.navigator.normalize_token(inner['register']).upper()})
            elif isinstance(part, list) and part:
                op = part[0]
                operators.append(op)
        if len(operators) == 1 and len(operands) >= 2:
            if operators[0] == '-':
                return {'subtract': operands[:2]}
            if operators[0] == '+':
                return {'add': operands[:2]}
        if operands:
            return {'add': operands}
        return None

    def _process_pseudo_value(self, value_data: List[Any]) -> Optional[Dict[str, Any]]:
        if not value_data or not isinstance(value_data[0], dict):
            return None
        atom = value_data[0].get('atom', [{}])[0]
        if 'integer' in atom:
            imm = self.registry.transform('integer', atom['integer'])
            return {'integer': {'type': imm.type, 'value': imm.value}} if imm else None
        if 'float_number' in atom:
            float_data = atom['float_number'][0]
            if float_data and len(float_data) >= 2:
                return {'float': {'type': float_data[1], 'value': float_data[0]}}
        if 'string' in atom:
            return {'string': atom['string'][0][0]} if atom['string'] else None
        if 'expression' in atom:
            return self._process_expression(atom)
        return None

    def _process_expression(self, expr_container: Dict[str, Any]) -> Any:
        visitor = ExpressionVisitor(self.navigator)
        return visitor.process(expr_container)

    # Small extraction helpers
    def _extract_section_name(self, directive_data: List[Any]) -> str:
        params = directive_data[1].get('section_params', [])
        if params and isinstance(params[0], dict):
            return self.navigator.normalize_token(params[0].get('name'))
        return ''

    def _extract_global_name(self, directive_data: List[Any]) -> str:
        params = directive_data[1].get('global_params', [])
        if params and isinstance(params[0], dict):
            return self.navigator.normalize_token(params[0].get('name'))
        return ''

# =============================================================================
# Deterministic AST -> legacy-dict serializer
# =============================================================================

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
        mem = {}
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


def _serialize_lgroup(lg: LabelGroup) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for it in lg.instructions:
        if isinstance(it, Label):
            items.append({'label': it.name})
        elif isinstance(it, Instruction):
            items.append({'instruction': _serialize_instruction(it)})
    return {'lgroup': items}


def ast_to_legacy_section_dict(section: Section) -> Dict[str, Any]:
    sec: Dict[str, Any] = {'name': section.name}
    if section.lgroups:
        sec['lgroups'] = [_serialize_lgroup(lg) for lg in section.lgroups]
    if section.pseudo_instruct:
        sec['pseudo_instruct'] = section.pseudo_instruct
    return sec


def ast_to_legacy_program_dict(program: Program) -> Dict[str, Any]:
    # Preserve legacy top-level shape: {"program": [ {"section": ...}, {"globals": [...] } ] }
    out: List[Any] = []
    # first entry: always the .text section (ensure it exists)
    # find .text in program.sections (there should be at least one)
    text_sec = None
    for s in program.sections:
        if s.name == '.text':
            text_sec = s
            break
    if text_sec is None and program.sections:
        text_sec = program.sections[0]
    if text_sec is None:
        text_sec = Section(name='.text')
    out.append({'section': ast_to_legacy_section_dict(text_sec)})

    # append globals store as second element (legacy behaviour)
    out.append({'globals': program.globals})

    # append other sections in order (excluding the already appended .text)
    for s in program.sections:
        if s.name == '.text':
            continue
        out.append({'section': ast_to_legacy_section_dict(s)})

    return {'program': out}

# =============================================================================
# Public API – transform now returns the legacy dict but uses typed AST
# =============================================================================
def transform(parse_tree: Dict[str, Any]) -> Dict[str, Any]:
    transformer = AsmTransformer()
    program_data = parse_tree.get('program', [])
    ast_root = transformer.visit_program(program_data, {})
    return ast_to_legacy_program_dict(ast_root)

# =============================================================================
# CLI driver
# =============================================================================
if __name__ == '__main__':
    try:
        parse_tree = json.load(sys.stdin)
        result = transform(parse_tree)
        json.dump(result, sys.stdout, indent=2)
    except json.JSONDecodeError:
        print('Error: Invalid JSON input.', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'Unexpected error: {e}', file=sys.stderr)
        sys.exit(1)
