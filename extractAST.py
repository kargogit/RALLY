"""
extractAST.py – Step 2 transformer that consumes a parse-tree
(dictionary-shaped, as produced by an ANTLR JSON exporter or similar)
and builds a typed AST using dataclasses from astNodes.py.

Public interface:
    transform(parse_tree: Dict[str, Any]) -> Dict[str, Any]
    (returns the legacy dict format via astNodes.ast_to_legacy_program_dict)
"""

from typing import Any, Dict, List, Optional, Callable
from abc import ABC
import json
import sys
import copy

# Import typed AST and serializer
from astNodes import (
    Program, Section, LabelGroup, Instruction, Label, Operand,
    Memory, Immediate, GlobalDecl, ast_to_legacy_program_dict
)


class TranslationError(Exception):
    """Raised when the AST extraction encounters a construct it explicitly does not support."""
    pass


# ---------------------------------------------------------------------------
# Simple token normalization / parse-tree navigation helper
# ---------------------------------------------------------------------------
class ParseTreeNavigator:
    @staticmethod
    def normalize_token(token: Any) -> str:
        """
        Robust token normalizer for a variety of parse-tree shapes.
        The parse trees coming from different ANTLR JSON exporters can vary,
        so this function attempts to extract a string token from common patterns.
        """
        if token is None:
            return ''
        if isinstance(token, str):
            return token
        if isinstance(token, (list, tuple)):
            if not token:
                return ''
            first = token[0]
            # nested common forms: [("NAME", ...)] or [('123', 'INT')]
            if isinstance(first, str):
                return first
            if isinstance(first, (list, tuple)) and first:
                # e.g. (value, type) pattern
                if isinstance(first[0], str):
                    return first[0]
                return str(first[0])
            # fallback to stringifying the first element
            return str(first)
        if isinstance(token, dict):
            # try common keys
            for key in ("name", "register", "size", "opcode", "dx", "terminator_opcode"):
                if key in token and token[key]:
                    return ParseTreeNavigator.normalize_token(token[key])
            # try first value
            for v in token.values():
                return ParseTreeNavigator.normalize_token(v)
        return str(token)


# ---------------------------------------------------------------------------
# Generic parse-tree visitor (double-dispatch by key)
# ---------------------------------------------------------------------------
class ParseTreeVisitor(ABC):
    def __init__(self):
        self.navigator = ParseTreeNavigator()

    def visit(self, node: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Inspects a dict node and dispatches to visit_<key> for the first matching key.
        Returns the result of that visitor or falls back to generic_visit.
        """
        if not isinstance(node, dict):
            return None
        context = context or {}
        prev_loc = context.get('current_loc')
        current_loc = node.get('_loc')

        if current_loc:
            context['current_loc'] = current_loc

        try:
            for key in node:
                if key == '_loc':
                    continue
                method_name = f"visit_{key}"
                if hasattr(self, method_name):
                    return getattr(self, method_name)(node[key], context)
            return self.generic_visit(node, context)
        finally:
            # Restore previous location (safe for nested nodes)
            if current_loc:
                if prev_loc is not None:
                    context['current_loc'] = prev_loc
                elif 'current_loc' in context:
                    del context['current_loc']

        for key in node.keys():
            method_name = f"visit_{key}"
            if hasattr(self, method_name):
                return getattr(self, method_name)(node[key], context)
        return self.generic_visit(node, context)

    def generic_visit(self, node: Any, context: Dict[str, Any]) -> Any:
        return node


# ---------------------------------------------------------------------------
# RIP-relative detection helper
# ---------------------------------------------------------------------------
class RipRelativeDetector:
    """
    Detects common RIP-relative forms in parse-tree operand fragments and
    returns a Memory dataclass with base='RIP' and a displacement (if found).
    This detector is conservative and focuses on extraction of symbolic
    displacements so that RIP-relative addressing is a first-class operand.
    """

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
            for k, v in node.items():
                if k in ('rel', 'rip', 'rel_prefix'):
                    return True
                if self._has_rip_or_rel(v):
                    return True
        elif isinstance(node, list):
            return any(self._has_rip_or_rel(item) for item in node)
        return False

    def _extract_rip_displacement(self, operand_data: Any) -> Optional[object]:
        # Best-effort: look for 'expression' containers or 'name' tokens inside lists/dicts
        if isinstance(operand_data, list):
            for item in operand_data:
                if isinstance(item, dict) and 'expression' in item:
                    expr = item['expression']
                    if isinstance(expr, list) and expr:
                        return self._extract_from_expression(expr[0])
                if isinstance(item, dict) and 'name' in item:
                    return self.navigator.normalize_token(item['name'])
        if isinstance(operand_data, dict):
            if 'name' in operand_data:
                return self.navigator.normalize_token(operand_data['name'])
        return None

    def _extract_from_expression(self, expr_data: Any) -> Optional[object]:
        if not isinstance(expr_data, dict):
            return None
        # Check for 'wrt' key
        if 'wrt' in expr_data:
            wrt_expr = expr_data['wrt']
            if isinstance(wrt_expr, list) and len(wrt_expr) >= 3:
                # Extract the symbol and the wrt target
                symbol_part = wrt_expr[0]
                wrt_target_part = wrt_expr[2]
                symbol = self.navigator.normalize_token(symbol_part)
                wrt_target = self.navigator.normalize_token(wrt_target_part)
                return f"{symbol} wrt {wrt_target}"
        # Continue with existing logic for other cases
        for key in ('additiveExpression', 'multiplicativeExpression', 'castExpression', 'unaryExpression'):
            if key in expr_data and expr_data[key]:
                elem = expr_data[key][0]
                # recurse
                if isinstance(elem, dict):
                    if 'name' in elem:
                        return self.navigator.normalize_token(elem['name'])
                    if 'integer' in elem:
                        int_item = elem['integer'][0]
                        if isinstance(int_item, (list, tuple)) and int_item:
                            return int_item[0]
                    # deeper recursion
                    res = self._extract_from_expression(elem)
                    if res is not None:
                        return res
        # fallback to top-level name/integer
        if 'name' in expr_data:
            return self.navigator.normalize_token(expr_data['name'])
        if 'integer' in expr_data:
            int_data = expr_data['integer'][0]
            if isinstance(int_data, (list, tuple)):
                return int_data[0]
        return None



# ---------------------------------------------------------------------------
# Transformation registry for small reusable handlers
# ---------------------------------------------------------------------------
class TransformationRegistry:
    def __init__(self):
        self._handlers = {}

    def register(self, node_type: str):
        def decorator(func: Callable):
            self._handlers[node_type] = func
            return func
        return decorator

    def transform(self, node_type: str, data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        handler = self._handlers.get(node_type)
        if handler:
            return handler(data, context or {})
        return None


# ---------------------------------------------------------------------------
# Expression visitor: lightweight expression extraction for immediates, names.
# ---------------------------------------------------------------------------
class ExpressionVisitor:
    def __init__(self, navigator: ParseTreeNavigator):
        self.navigator = navigator

    def process(self, expr_container: Dict[str, Any]) -> Any:
        if not expr_container or 'expression' not in expr_container:
            return None
        expr_list = expr_container['expression']
        if not expr_list or not isinstance(expr_list[0], dict):
            return None
        actual_expr = expr_list[0]
        # simple patterns handled: castExpression, additiveExpression, multiplicativeExpression
        if 'castExpression' in actual_expr:
            return self._visit_cast(actual_expr['castExpression'])
        if 'additiveExpression' in actual_expr:
            return self._visit_additive(actual_expr['additiveExpression'])
        if 'multiplicativeExpression' in actual_expr:
            return self._visit_multiplicative(actual_expr['multiplicativeExpression'])

        # STRICT CHECK: If we are here, the expression contains something we don't know how to handle
        # e.g., 'wrt', 'seg', 'binaryExpression' with unknown ops
        keys = list(actual_expr.keys())
        if keys and keys != ['_loc']:
            raise TranslationError(f"Unsupported expression construct(s): {keys}")

        return None

    def _visit_cast(self, cast_expr: List[Any]) -> Any:
        expr = cast_expr[0]
        if 'name' in expr:
            return self.navigator.normalize_token(expr['name'])
        if 'register' in expr:
            return {'register': self.navigator.normalize_token(expr['register']).upper()}
        if 'integer' in expr:
            val = expr['integer'][0]
            # val typically like (value, type) and may include ascii as a 3rd element
            int_dict = {'type': val[1], 'value': val[0]}
            if len(val) >= 3:
                int_dict['ascii'] = val[2]
            return {'integer': int_dict}
        if 'unaryExpression' in expr:
            return self._visit_unary(expr['unaryExpression'])

        # Strict check for cast contents
        if isinstance(expr, dict):
            known = {'name', 'register', 'integer', 'unaryExpression', '_loc'}
            unknown = set(expr.keys()) - known
            if unknown:
                raise TranslationError(f"Unsupported cast expression content: {unknown}")

        return None

    def _visit_unary(self, unary_expr: List[Any]) -> Any:
        if len(unary_expr) == 2:
            op_token = unary_expr[0].get('unaryOperator', '')
            op = self.navigator.normalize_token(op_token)
            operand = self.process({'expression': [unary_expr[1]]})
            if isinstance(operand, dict) and 'integer' in operand:
                # make sure integer value is signed if unary contains '-'
                try:
                    operand_value = int(operand['integer']['value'])
                    if op == '-':
                        operand['integer']['value'] = -operand_value
                except Exception:
                    # keep original if conversion fails
                    pass
                return operand
            return {'unary_op': op, 'unary_val': operand}
        return None

    def _visit_additive(self, add_expr: List[Any]) -> Any:
        operands = []
        for comp in add_expr:
            if isinstance(comp, dict):
                if 'multiplicativeExpression' in comp or 'castExpression' in comp:
                    res = self.process({'expression': [comp]})
                    operands.append(res)
        if not operands:
            return None
        return {'additive': operands} if len(operands) > 1 else operands[0]

    def _visit_multiplicative(self, mul_expr: List[Any]) -> Any:
        operands = []
        for comp in mul_expr:
            if isinstance(comp, dict) and 'castExpression' in comp:
                operands.append(self.process({'expression': [comp]}))
        if not operands:
            return None
        return operands if len(operands) > 1 else operands[0]


# ---------------------------------------------------------------------------
# Concrete transformer: builds typed dataclasses and uses serializer for output
# ---------------------------------------------------------------------------
class AsmTransformer(ParseTreeVisitor):
    def __init__(self):
        super().__init__()
        self.registry = TransformationRegistry()
        self.rip_detector = RipRelativeDetector(self.navigator)
        self._setup_handlers()
        # default text section present to preserve legacy behavior
        self.text_section = Section(name='.text')

    def _setup_handlers(self):
        @self.registry.register('integer')
        def handle_integer(data, ctx):
            if isinstance(data, list) and data:
                int_data = data[0]
                if isinstance(int_data, (list, tuple)) and len(int_data) >= 2:
                    # int_data is typically (value, type) but may carry a 3rd element for ascii
                    ascii_ch = int_data[2] if len(int_data) >= 3 else None
                    return Immediate(value=int_data[0], type=int_data[1], ascii=ascii_ch)
            return None

        @self.registry.register('string')
        def handle_string(data, ctx):
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, (list, tuple)) and len(first) >= 2:
                    string_val = first[0]
                    # Handle single-character literals like "'-'"
                    if len(string_val) == 3 and string_val[0] == "'" and string_val[-1] == "'":
                        char_val = string_val[1]
                        return Immediate(value=ord(char_val), type='BYTE', ascii=char_val)
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


    def _convert_expression_to_memory(self, expr: Any) -> Optional[Memory]:
        """
        Convert simple address expressions (common in memory operands) to Memory nodes.
        Used only for non-LEA instructions that perform actual memory access.
        Handles:
        - Single register: {"register": "RSI"} → base="RSI"
        - Register + displacement: additive [reg, int] → base=reg, displacement=int
        - Symbolic name (under default rel): "sym" → base="RIP", displacement="sym"
        Returns None for complex cases (e.g., two registers without scale) → keeps expression.
        """
        if expr is None:
            return None

        mem = Memory()

        # Symbolic name → RIP-relative (valid under default rel)
        if isinstance(expr, str):
            mem.base = "RIP"
            mem.displacement = expr
            return mem

        if not isinstance(expr, dict):
            return None

        # Single register
        if "register" in expr and len(expr) == 1:
            mem.base = expr["register"]
            return mem

        # Additive expression: expect exactly two parts (reg + disp or disp + reg)
        if "additive" in expr and isinstance(expr["additive"], list) and len(expr["additive"]) == 2:
            addends = expr["additive"]
            reg = None
            disp = None
            for part in addends:
                if isinstance(part, dict):
                    if "register" in part:
                        if reg is not None:
                            return None  # two registers → too complex, keep as expression
                        reg = part["register"]
                    elif "integer" in part:
                        if disp is not None:
                            return None
                        disp_val = part["integer"]["value"]
                        try:
                            disp = int(disp_val, 0)  # handles hex if needed
                        except ValueError:
                            disp = disp_val
            if reg:
                mem.base = reg
                if disp is not None:
                    mem.displacement = disp
                return mem

        # Anything else (e.g., reg + reg, scaled index, complex expr) → keep as expression
        return None


    # Top-level program builder
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
                # do not duplicate .text section
                if result.name != '.text':
                    prog.sections.append(result)
                context['current_section'] = result
            elif isinstance(result, GlobalDecl):
                prog.globals.append(result.name)
            elif isinstance(result, LabelGroup):
                ctx_sec: Section = context['current_section']
                ctx_sec.children.append(result)
            elif isinstance(result, dict) and 'pseudo_instruct' in result:
                # --- MERGE-CONTINUATION FIX ---
                ctx_sec: Section = context['current_section']
                pseudo = result['pseudo_instruct']
                # only merge when the pseudo has neither 'directive' nor 'name'
                # (this indicates a continuation of the previous pseudo-instruction)
                if 'directive' not in pseudo and 'name' not in pseudo and ctx_sec.pseudo_instruct:
                    last = ctx_sec.pseudo_instruct[-1]
                    # If current has a 'dx' and last doesn't, adopt it; otherwise keep last's dx.
                    if 'dx' in pseudo and 'dx' not in last:
                        last['dx'] = pseudo['dx']
                    # Merge 'values' arrays
                    if 'values' in pseudo:
                        last_values = last.get('values', [])
                        last_values.extend(pseudo['values'])
                        last['values'] = last_values
                    # Merge 'params' if present (some pseudo-instructions may carry params)
                    if 'params' in pseudo:
                        last_params = last.get('params', [])
                        last_params.extend(pseudo['params'])
                        last['params'] = last_params
                    # Merge other small fields conservatively (e.g., integer)
                    for k in ('integer', 'equ', 'resx'):
                        if k in pseudo and k not in last:
                            last[k] = pseudo[k]

                else:
                    # normal case: a standalone pseudo-instruction with a name (or no previous)
                    ctx_sec.pseudo_instruct.append(pseudo)

        return prog

    # generic line visitor that handles directive/pseudoinstruction etc.
    def visit_line(self, line_data: List[Any], context: Dict[str, Any]) -> Optional[Any]:
        if not line_data or not isinstance(line_data[0], dict):
            return None
        line_content = line_data[0]
        if 'directive' in line_content:
            return self._process_directive(line_content['directive'], context)
        if 'pseudoinstruction' in line_content:
            return self._process_pseudoinstruction(line_content['pseudoinstruction'], context)

        # Strict Check: If line contains keys we don't recognize
        known_keys = {'directive', 'pseudoinstruction', 'instruction', 'label', '_loc'}
        unknowns = set(line_content.keys()) - known_keys
        if unknowns:
            raise TranslationError(f"Unknown line type: {unknowns}")

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
                if not line_node or not isinstance(line_node, list) or not line_node[0]:
                    continue
                instr_data = (line_node[0].get('instruction') or line_node[0].get('terminator_instruction'))
                if instr_data:
                    # Prefer the explicit _loc present on the wrapper node (item or line_node[0])
                    prev_loc = context.get('current_loc')
                    new_loc = None
                    # check the outer item first (some exporters put _loc there)
                    if isinstance(item, dict) and '_loc' in item:
                        new_loc = item['_loc']
                    # otherwise prefer the line wrapper's _loc if present
                    elif isinstance(line_node[0], dict) and '_loc' in line_node[0]:
                        new_loc = line_node[0]['_loc']

                    if new_loc is not None:
                        context['current_loc'] = new_loc

                    try:
                        instr_node = self.visit_instruction(instr_data, context)
                    finally:
                        # restore previous location (or remove if none)
                        if prev_loc is not None:
                            context['current_loc'] = prev_loc
                        elif 'current_loc' in context:
                            del context['current_loc']

                    if instr_node:
                        lg.instructions.append(instr_node)

            else:
                 # Check for unhandled block items
                known_keys = {'label', 'non_terminator_line', 'terminator_line', '_loc'}
                unknowns = set(item.keys()) - known_keys
                if unknowns:
                     # This often catches things like 'directive' inside a block where only instructions were expected
                     raise TranslationError(f"Unexpected item in block: {unknowns}")

        return lg if lg.instructions else None

    def visit_label(self, label_data: List[Any], context: Dict[str, Any]) -> Optional[Label]:
        if not label_data or not isinstance(label_data[0], dict):
            return None
        name = self.navigator.normalize_token(label_data[0].get('name'))
        label = Label(name=name)
        if context.get('current_loc'):
            label.location = context['current_loc']
        return label

    def visit_instruction(self, instr_data: List[Any], context: Dict[str, Any]) -> Optional[Instruction]:
        instr = Instruction(opcode='')

        # If instr_data is a dict wrapper that contains its own _loc, prefer that.
        # Otherwise fall back to context['current_loc'] (set by wrapper nodes).
        explicit_loc = None
        if isinstance(instr_data, dict) and '_loc' in instr_data:
            explicit_loc = instr_data['_loc']
        elif isinstance(instr_data, list) and instr_data:
            # sometimes the wrapper is a list containing a dict with _loc
            first = instr_data[0]
            if isinstance(first, dict) and '_loc' in first:
                explicit_loc = first['_loc']

        loc = explicit_loc if explicit_loc is not None else context.get('current_loc')
        if loc:
            # copy so multiple AST nodes don't share the same mutable dict
            instr.location = copy.deepcopy(loc)
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
            elif '_loc' in piece:
                pass
            else:
                raise TranslationError(f"Unhandled instruction component: {piece.keys()}")
    # === Post-process memory operands for actual memory-accessing instructions ===
        if instr.opcode != "LEA":
            for op in instr.operands:
                if op.expression is not None and op.memory is None:
                    converted = self._convert_expression_to_memory(op.expression)
                    if converted is not None:
                        op.memory = converted
                        op.expression = None
                    # If not converted, leave as expression (safety for unexpected cases)

        return instr if instr.opcode else None

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
                # Evaluate expression; if it reduces to a plain integer immediate,
                # store it in op.integer for consistency with direct 'integer' operands.
                expr_res = self._process_expression(item)
                if isinstance(expr_res, dict) and 'integer' in expr_res and len(expr_res) == 1:
                    int_rec = expr_res['integer']
                    # propagate ascii if present in the expression result
                    op.integer = Immediate(value=int_rec['value'],
                                        type=int_rec['type'],
                                        ascii=int_rec.get('ascii'))
                else:
                    # Non-trivial expression: keep as expression field
                    op.expression = expr_res
            elif 'integer' in item:
                imm_node = self.registry.transform('integer', item['integer'])
                if imm_node:
                    op.integer = imm_node
            elif 'name' in item:
                op.name = self.navigator.normalize_token(item['name'])
            elif 'string' in item:
                string_node = self.registry.transform('string', item['string'])
                if string_node:
                    op.integer = string_node
            else:
                # STRICT CHECK: Unhandled operand keys (e.g. 'segment', 'wrt', complex types)
                known_keys = {'register', 'size', 'expression', 'integer', 'name', 'string', '_loc', 'rel_prefix'}
                unknowns = set(item.keys()) - known_keys
                if unknowns:
                    raise TranslationError(f"Unhandled operand component: {unknowns}")
        return op


    def _process_directive(self, directive_data: List[Any], context: Dict[str, Any]) -> Optional[object]:
        """
        Robust directive handler.
        """
        if not directive_data:
            return None

        # Helper: extract a normalized token from a possibly-nested value
        def _extract_token(val: Any) -> Optional[str]:
            if val is None:
                return None
            if isinstance(val, (list, tuple)):
                if not val:
                    return None
                first = val[0]
                if isinstance(first, (list, tuple)) and first:
                    return first[0]
                return self.navigator.normalize_token(val)
            if isinstance(val, dict):
                for vv in val.values():
                    tok = _extract_token(vv)
                    if tok is not None:
                        return tok
            return self.navigator.normalize_token(val)

        # Determine directive name
        first = directive_data[0]
        if isinstance(first, dict):
            try:
                directive_name = next(iter(first.keys()))
            except StopIteration:
                directive_name = self.navigator.normalize_token(first)
        else:
            directive_name = self.navigator.normalize_token(first)

        # Special built-in handling
        if directive_name == 'section':
            name = self._extract_section_name(directive_data)
            if name == '.text':
                return self.text_section
            new_section = Section(name=name)
            if context.get('current_loc'):
                new_section.location = context['current_loc']
            return new_section

        if directive_name == 'global':
            name = self._extract_global_name(directive_data)
            decl = GlobalDecl(name=name)
            if context.get('current_loc'):
                decl.location = context['current_loc']
            return decl

        # === NEW: Special handling for multi-symbol extern ===
        if directive_name == 'extern':
            params: List[str] = []

            def collect_names(node: Any) -> None:
                if isinstance(node, list):
                    for elem in node:
                        if isinstance(elem, dict):
                            if 'name' in elem:
                                name = self.navigator.normalize_token(elem['name'])
                                if name:
                                    params.append(name)
                            elif 'extern_params' in elem:
                                collect_names(elem['extern_params'])
                elif isinstance(node, dict):
                    if 'name' in node:
                        name = self.navigator.normalize_token(node['name'])
                        if name:
                            params.append(name)
                    elif 'extern_params' in node:
                        collect_names(node['extern_params'])

            # The parameter list is in directive_data[1]['extern_params'] (if present)
            if len(directive_data) >= 2:
                container = directive_data[1]
                if isinstance(container, dict) and 'extern_params' in container:
                    collect_names(container['extern_params'])

            pseudo = {'directive': 'extern'}
            if params:
                pseudo['params'] = params
            return {'pseudo_instruct': pseudo}

        # === Generic fallback for everything else (default rel, align, etc.) ===
        params: List[str] = []
        seen = set()

        def add_param(tok: Optional[str]) -> None:
            if tok is None:
                return
            if tok not in seen:
                seen.add(tok)
                params.append(tok)

        # Extract tokens once per item (avoid double-extraction)
        for item in directive_data[1:]:
            tok = _extract_token(item)
            add_param(tok)

        # If no params found yet, try pulling from the directive keyword node itself
        if not params and isinstance(first, dict):
            for v in first.values():
                tok = _extract_token(v)
                add_param(tok)

        pseudo = {'directive': directive_name}
        if params:
            pseudo['params'] = params

        return {'pseudo_instruct': pseudo}

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
                else:
                    # Strict check: item['value'] existed but produced no result
                    raise TranslationError(f"Unsupported value definition in pseudo-instruction: {item['value']}")
            elif 'integer' in item:
                imm = self.registry.transform('integer', item['integer'])
                if imm:
                    int_entry = {'type': imm.type, 'value': imm.value}
                    if getattr(imm, 'ascii', None) is not None:
                        int_entry['ascii'] = imm.ascii
                    pseudo_dict['integer'] = int_entry
            elif '_loc' in item:
                pass
            else:
                 raise TranslationError(f"Unsupported pseudo-instruction key: {item.keys()}")
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
            if not imm:
                return None
            int_entry = {'type': imm.type, 'value': imm.value}
            if getattr(imm, 'ascii', None) is not None:
                int_entry['ascii'] = imm.ascii
            return {'integer': int_entry}
        if 'float_number' in atom:
            float_data = atom['float_number'][0]
            if float_data and len(float_data) >= 2:
                return {'float': {'type': float_data[1], 'value': float_data[0]}}
        if 'string' in atom:
            return {'string': atom['string'][0][0]} if atom['string'] else None
        if 'expression' in atom:
            return self._process_expression(atom)
        # STRICT CHECK: Handle bare symbols/names inside pseudo values (common in dq/dd)
        if 'name' in atom:
             # Just normalize the name and treat as a symbol reference
             return {'symbol': self.navigator.normalize_token(atom['name'])}

        # If we got here, it's an atom type we don't know (e.g. char constant not in string/int form)
        raise TranslationError(f"Unsupported pseudo-value atom: {atom.keys()}")

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


# ---------------------------------------------------------------------------
# Public API: transform(parse_tree) -> legacy dict (uses typed AST internally)
# ---------------------------------------------------------------------------
def transform(parse_tree: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build typed AST from parse_tree and return legacy dict form.
    parse_tree is expected to be a dict with key 'program' mapping to a list.
    """
    transformer = AsmTransformer()
    program_data = parse_tree.get('program', [])
    ast_root = transformer.visit_program(program_data, {})
    return ast_to_legacy_program_dict(ast_root)


# ---------------------------------------------------------------------------
# CLI driver (JSON in / JSON out)
# ---------------------------------------------------------------------------
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
