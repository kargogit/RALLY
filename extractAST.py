"""
asm_transformer.py – Step 2 of the NASM → LLVM‑IR pipeline

This module implements the **AST construction** phase described in the
Plan‑of‑Action Step 2.  An ANTLR parse tree (produced from the
context‑free NASM grammar of Step 1) is visited and transformed into a
hierarchical, dictionary‑based abstract syntax tree (AST) that is free
from syntactic noise but retains all information required for PIC‑aware
LLVM‑IR generation (e.g. RIP‑relative operands, symbolic labels).
"""
# =============================================================================
# Imports
# =============================================================================
from typing import Any, Dict, List, Optional, Union, Callable, Protocol
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from collections import defaultdict
import json
import sys

# =============================================================================
# Visitor protocol – defines the contract for concrete AST visitors
# =============================================================================
class ASTVisitor(Protocol):
    """
    A thin protocol that enumerates the explicit ``visit_*`` methods
    required by the transformation step.  Using a protocol (instead of a
    concrete base class) keeps the implementation flexible while still
    providing static type checking for callers.
    """
    def visit_program(self, node: Any, context: Any) -> Any: ...
    def visit_section(self, node: Any, context: Any) -> Any: ...
    def visit_lgroup(self, node: Any, context: Any) -> Any: ...
    def visit_instruction(self, node: Any, context: Any) -> Any: ...
    def visit_operand(self, node: Any, context: Any) -> Any: ...
    def visit_label(self, node: Any, context: Any) -> Any: ...
    def generic_visit(self, node: Any, context: Any) -> Any: ...

# =============================================================================
# Immutable AST node definitions (dictionary‑serialisable via ``asdict``)
# =============================================================================
@dataclass
class ASTNode:
    """Base class – exists solely for type hierarchy clarity."""
    pass

@dataclass
class Program(ASTNode):
    """Root node – contains a list of sections."""
    sections: List['Section'] = field(default_factory=list)

@dataclass
class Section(ASTNode):
    """Logical assembly section (e.g. ``.text`` or ``.data``)."""
    name: str
    lgroups: List['LabelGroup'] = field(default_factory=list)
    pseudo_instruct: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class LabelGroup(ASTNode):
    """A contiguous block of labels and instructions."""
    instructions: List[Union['Instruction', 'Label']] = field(default_factory=list)

@dataclass
class Instruction(ASTNode):
    """Opcode together with an ordered list of operand descriptors."""
    opcode: str
    operands: List['Operand'] = field(default_factory=list)
    prefix: Optional[str] = None   # e.g. LOCK, REP

@dataclass
class Label(ASTNode):
    """Symbolic identifier used for jumps, calls and data references."""
    name: str

@dataclass
class Operand(ASTNode):
    """Unified container – exactly one of the optional fields will be set."""
    register: Optional[str] = None
    memory: Optional['Memory'] = None
    immediate: Optional['Immediate'] = None
    expression: Optional['Expression'] = None
    size: Optional[str] = None
    name: Optional[str] = None      # for plain identifiers

@dataclass
class Memory(ASTNode):
    """Memory operand shape – covers `[base + index*scale + disp]` and RIP‑relative."""
    base: Optional[str] = None
    index: Optional[str] = None
    scale: Optional[int] = None
    displacement: Optional[Union[int, str]] = None

@dataclass
class Immediate(ASTNode):
    """Concrete immediate value (numeric or symbolic)."""
    value: Union[int, str]
    type: str                       # e.g. ``byte``, ``dword``

@dataclass
class Register(ASTNode):
    name: str

@dataclass
class Name(ASTNode):
    value: str

@dataclass
class Expression(ASTNode):
    """Tree node for parsed expressions (additive, multiplicative, etc.)."""
    type: str
    operands: List[ASTNode] = field(default_factory=list)
    operator: Optional[str] = None

# =============================================================================
# Helper utilities – parsing‑tree navigation
# =============================================================================
class ParseTreeNavigator:
    """
    Small collection of functions that know how to extract a normalized
    textual representation from the heterogeneous token structures emitted
    by the ANTLR parser.  Centralising this logic avoids duplication
    across the many visitor methods.
    """
    @staticmethod
    def normalize_token(token: Any) -> str:
        """
        Convert the various token representations (string, tuple, list,
        dict) into a plain ``str``.  The implementation tolerates the
        different shapes that appear in the generated parse tree.
        """
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
# Generic parse‑tree visitor – provides the double‑dispatch mechanism
# =============================================================================
class ParseTreeVisitor(ABC):
    """
    Base class implementing a ``visit`` method that performs a simple
    key‑based dispatch.  Concrete visitors (e.g. ``AsmTransformer``)
    inherit from this class and provide ``visit_<key>`` methods for the
    parse‑tree nodes they care about.
    """
    def __init__(self):
        self.navigator = ParseTreeNavigator()

    def visit(self, node: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """Dispatch to a ``visit_<key>`` method if one matches the node."""
        if not isinstance(node, dict):
            return None
        context = context or {}

        # The first key that has an associated method wins – this mirrors the
        # structure of the ANTLR grammar where each alternative is a distinct
        # key in the parse‑tree dict.
        for key in node.keys():
            method_name = f"visit_{key}"
            if hasattr(self, method_name):
                return getattr(self, method_name)(node[key], context)

        # Fallback – callers may rely on ``generic_visit`` to handle unknowns.
        return self.generic_visit(node, context)

    def generic_visit(self, node: Any, context: Dict[str, Any]) -> Any:
        """Default behaviour – return the raw node unmodified."""
        return node

# =============================================================================
# RIP‑relative detection – isolates PIC‑specific logic
# =============================================================================
class RipRelativeDetector:
    """
    Dedicated helper that recognises the ``[rel label]`` and
    ``[rip + offset]`` forms and synthesises a :class:`Memory` node with
    ``base="RIP"``.  Keeping this logic separate from the generic operand
    visitor respects the Single‑Responsibility Principle and makes the
    transformation easier to test in isolation.
    """
    def __init__(self, navigator: ParseTreeNavigator):
        self.navigator = navigator

    def detect(self, operand_data: Any) -> Optional[Memory]:
        """Return a ``Memory`` node if the operand is RIP‑relative, else ``None``."""
        if not operand_data:
            return None

        if self._has_rip_or_rel(operand_data):
            displacement = self._extract_rip_displacement(operand_data)
            return Memory(base="RIP", displacement=displacement)
        return None

    # --------------------------------------------------------------
    # Private helpers – recursive tree walkers
    # --------------------------------------------------------------
    def _has_rip_or_rel(self, node: Any) -> bool:
        """True if any descendant key is ``'rip'`` or ``'rel'``."""
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
        """
        Pull the symbolic or numeric displacement that appears together with
        the RIP base.  The parse‑tree format for a ``[rel label]`` construct
        places the identifier inside an ``'expression'`` list – we walk that
        structure until we find a ``name`` or ``integer`` token.
        """
        if isinstance(operand_data, list):
            for item in operand_data:
                if isinstance(item, dict) and 'expression' in item:
                    expr = item['expression']
                    if isinstance(expr, list) and expr:
                        return self._extract_from_expression(expr[0])
        return None

    def _extract_from_expression(self, expr_data: Any) -> Optional[Union[int, str]]:
        """Recursive extractor used by ``_extract_rip_displacement``."""
        if not isinstance(expr_data, dict):
            return None

        # Additive chains (e.g., rel label + 4) – walk each part.
        if 'additiveExpression' in expr_data:
            add_expr = expr_data['additiveExpression']
            if len(add_expr) > 1:
                for part in add_expr:
                    if isinstance(part, dict):
                        res = self._extract_from_expression(part)
                        if res:
                            return res

        # Multiplicative, cast and unary layers – descend further.
        if 'multiplicativeExpression' in expr_data:
            return self._extract_from_expression(expr_data['multiplicativeExpression'][0])
        if 'castExpression' in expr_data:
            return self._extract_from_expression(expr_data['castExpression'][0])
        if 'unaryExpression' in expr_data and len(expr_data['unaryExpression']) > 1:
            return self._extract_from_expression(expr_data['unaryExpression'][1])

        # Base cases – name or integer token.
        if 'name' in expr_data:
            return self.navigator.normalize_token(expr_data['name'])
        if 'integer' in expr_data:
            int_data = expr_data['integer'][0]
            if isinstance(int_data, (list, tuple)) and len(int_data) == 2:
                return int_data[0]

        return None

# =============================================================================
# Transformation registry – decouples node‑type handling from visitor logic
# =============================================================================
class TransformationRegistry:
    """
    Holds a mapping ``node_type -> handler``.  Handlers are registered
    via the ``@registry.register("<type>")`` decorator.  This pattern
    enables extension points without touching the core ``AsmTransformer``
    code: adding support for a new parse‑tree node is simply a matter of
    installing another handler.
    """
    def __init__(self):
        self._handlers: Dict[str, Callable] = {}

    def register(self, node_type: str):
        """Decorator that stores *func* under *node_type*."""
        def decorator(func: Callable):
            self._handlers[node_type] = func
            return func
        return decorator

    def transform(self, node_type: str, data: Any,
                  context: Optional[Dict[str, Any]] = None) -> Any:
        """Invoke the handler for *node_type* if it exists."""
        handler = self._handlers.get(node_type)
        if handler:
            return handler(data, context or {})
        return None

# =============================================================================
# Concrete visitor – builds the dictionary‑based AST
# =============================================================================
class AsmTransformer(ParseTreeVisitor):
    """
    Traverses the ANTLR parse tree and emits a JSON‑serialisable AST that
    matches the structure laid out in Step 2 of the plan.  The transformer
    is deliberately side‑effect free: it constructs new dicts/lists and
    never mutates an already‑emitted node.
    """
    def __init__(self):
        super().__init__()
        self.registry = TransformationRegistry()
        self.rip_detector = RipRelativeDetector(self.navigator)
        self._setup_handlers()

        # The ``.text`` section is always present; we construct it once
        # and reuse the same object for any subsequent ``.text`` directives.
        self.text_section = Section(name=".text")

    # -----------------------------------------------------------------
    # Handler registration – maps parse‑tree leaf types to small helpers
    # -----------------------------------------------------------------
    def _setup_handlers(self):
        @self.registry.register("integer")
        def handle_integer(data, ctx):
            """Turn a raw ``integer`` token into an :class:`Immediate`."""
            if isinstance(data, list) and data:
                int_data = data[0]
                if isinstance(int_data, (list, tuple)) and len(int_data) == 2:
                    return Immediate(value=int_data[0], type=int_data[1])
            return None

        @self.registry.register("label")
        def handle_label(data, ctx):
            """Delegate to the explicit ``visit_label`` method."""
            return self.visit_label(data, ctx)

        @self.registry.register("operand")
        def handle_operand(data, ctx):
            """Delegate to the explicit ``visit_operand`` method."""
            return self.visit_operand(data, ctx)

        @self.registry.register("instruction")
        def handle_instruction(data, ctx):
            """Delegate to the explicit ``visit_instruction`` method."""
            return self.visit_instruction(data, ctx)

    # -----------------------------------------------------------------
    # Program entry point – assembles sections and globals
    # -----------------------------------------------------------------
    def visit_program(self,
                      program_data: List[Any],
                      context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert the top‑level ``program`` list into a dict with a ``program``
        key that holds an ordered list of sections and a global‑symbol table.
        """
        output: List[Dict[str, Any]] = []

        # Pre‑seed the output with the default ``.text`` section.
        text_section_dict = self._section_to_dict(self.text_section)
        global_store = {"globals": []}
        output.append({"section": text_section_dict})
        output.append(global_store)

        # Remember where new nodes should be attached.
        context["current_section_dict"] = output[0]

        for node in program_data:
            if not isinstance(node, dict):
                continue

            result = self.visit(node, context)
            if not result:
                continue

            # Dispatch based on the shape of ``result``.
            if "section" in result:
                # New section – replace the current pointer unless it is the
                # already‑initialised ``.text``.
                if result["section"]["name"] != ".text":
                    output.append(result)
                context["current_section_dict"] = result
            elif "global" in result:
                global_store["globals"].append(result["global"])
            elif "lgroup" in result:
                context["current_section_dict"]["section"]["lgroups"].append(result)
            elif "pseudo_instruct" in result:
                ctx_sec = context["current_section_dict"]["section"]
                ctx_sec["pseudo_instruct"].append(result["pseudo_instruct"])

        return {"program": output}

    # -----------------------------------------------------------------
    # Line handling – distinguishes directives from pseudo‑instructions
    # -----------------------------------------------------------------
    def visit_line(self,
                   line_data: List[Any],
                   context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """A line is either a directive, a pseudo‑instruction or an empty line."""
        if not line_data or not isinstance(line_data[0], dict):
            return None

        line_content = line_data[0]
        if "directive" in line_content:
            return self._process_directive(line_content["directive"], context)
        if "pseudoinstruction" in line_content:
            return self._process_pseudoinstruction(line_content["pseudoinstruction"], context)
        return None

    # -----------------------------------------------------------------
    # Logical block handling – a sequence of labels and instructions
    # -----------------------------------------------------------------
    def visit_block(self,
                    lgroup_data: List[Any],
                    context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Collect all labels and instructions belonging to a block."""
        lgroup_nodes: List[Dict[str, Any]] = []

        for item in lgroup_data:
            if not isinstance(item, dict):
                continue

            if "label" in item:
                label_node = self.visit_label(item["label"], context)
                if label_node:
                    lgroup_nodes.append(label_node)
            elif "non_terminator_line" in item or "terminator_line" in item:
                line_node = item.get("non_terminator_line") or item.get("terminator_line")
                # The ``instruction`` key differs between normal and terminator lines.
                instr_data = (line_node[0].get("instruction") or
                              line_node[0].get("terminator_instruction"))
                if instr_data:
                    instr_node = self.visit_instruction(instr_data, context)
                    if instr_node:
                        lgroup_nodes.append(instr_node)

        return {"lgroup": lgroup_nodes} if lgroup_nodes else None

    # -----------------------------------------------------------------
    # Primitive visitors – labelled, instruction and operand
    # -----------------------------------------------------------------
    def visit_label(self,
                    label_data: List[Any],
                    context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract the label name and wrap it in a dict."""
        if not label_data or not isinstance(label_data[0], dict):
            return None
        name = self.navigator.normalize_token(label_data[0].get("name"))
        return {"label": name}

    def visit_instruction(self,
                          instr_data: List[Any],
                          context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build an :class:`Instruction` object from its constituent pieces."""
        instruction = Instruction(opcode="")

        for piece in instr_data:
            if not isinstance(piece, dict):
                continue

            if "lock_prefix" in piece:
                instruction.prefix = "LOCK"
            elif "opcode" in piece or "terminator_opcode" in piece:
                key = "opcode" if "opcode" in piece else "terminator_opcode"
                instruction.opcode = self.navigator.normalize_token(piece[key]).upper()
            elif "operand" in piece:
                operand = self.visit_operand(piece["operand"], context)
                instruction.operands.append(operand)

        return {"instruction": self._instruction_to_dict(instruction)}

    def visit_operand(self,
                      operand_data: List[Any],
                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Translate a raw operand into a structured dictionary."""
        operand_dict: Dict[str, Any] = {}

        # 1️⃣  Detect RIP‑relative addressing first – this takes precedence
        #    over ordinary register/memory handling.
        rip_memory = self.rip_detector.detect(operand_data)
        if rip_memory:
            operand_dict["memory"] = asdict(rip_memory)
            return operand_dict

        # 2️⃣  Normal operand processing.
        for item in operand_data:
            if not isinstance(item, dict):
                continue

            if "register" in item:
                operand_dict["register"] = self.navigator.normalize_token(item["register"]).upper()
            elif "size" in item:
                operand_dict["size"] = self.navigator.normalize_token(item["size"]).upper()
            elif "expression" in item:
                operand_dict["expression"] = self._process_expression(item)
            elif "integer" in item:
                imm_node = self.registry.transform("integer", item["integer"])
                if imm_node:
                    operand_dict["integer"] = {"value": imm_node.value, "type": imm_node.type}
            elif "name" in item:
                operand_dict["name"] = self.navigator.normalize_token(item["name"])

        return operand_dict

    # -----------------------------------------------------------------
    # Directive and pseudo‑instruction processing helpers
    # -----------------------------------------------------------------
    def _process_directive(self,
                           directive_data: List[Any],
                           context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle ``section`` and ``global`` directives; ignore others."""
        if not directive_data or len(directive_data) < 2:
            return None

        directive_type = directive_data[0]

        if "section" in directive_type:
            name = self._extract_section_name(directive_data)
            if name == ".text":
                return {"section": self._section_to_dict(self.text_section)}
            # Create a fresh Section for any non‑text segment.
            new_section = Section(name=name)
            return {"section": self._section_to_dict(new_section)}

        if "global" in directive_type:
            name = self._extract_global_name(directive_data)
            return {"global": name}

        return None

    def _process_pseudoinstruction(self,
                                   pseudo_data: List[Any],
                                   context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Translate NASM pseudo-instructions (e.g. ``db``, ``dd``, ``dq``) into
        a small dictionary that records the name and any operand values.
        """
        pseudo_dict: Dict[str, Any] = {}
        pseudo_values: List[Any] = []

        equ_pending = False  # <-- new: remember if we saw an `equ` token

        for item in pseudo_data:
            if not isinstance(item, dict):
                continue

            if "name" in item:
                pseudo_dict["name"] = self.navigator.normalize_token(item["name"])
            elif "dx" in item:
                pseudo_dict["dx"] = self.navigator.normalize_token(item["dx"])
            elif "resx" in item:
                pseudo_dict["resx"] = self.navigator.normalize_token(item["resx"])
            elif "equ" in item:
                # mark that the next `expression` belongs to the `equ` construct
                equ_pending = True
            elif "expression" in item and equ_pending:
                # build the special `equ` expression shape and attach it
                equ_expr = self._build_equ_expression(item["expression"])
                if equ_expr:
                    pseudo_dict["equ"] = {"expression": equ_expr}
                equ_pending = False
            elif "value" in item:
                val = self._process_pseudo_value(item["value"])
                if val:
                    pseudo_values.append(val)
            elif "integer" in item:
                imm = self.registry.transform("integer", item["integer"])
                if imm:
                    pseudo_dict["integer"] = {"type": imm.type, "value": imm.value}

        if pseudo_values:
            pseudo_dict["values"] = pseudo_values
        return {"pseudo_instruct": pseudo_dict} if pseudo_dict else None

    def _build_equ_expression(self, expr_container: List[Any]) -> Optional[Dict[str, Any]]:
        """
        Special-case builder for `equ` expressions.  Handles simple additive
        forms like:  `$ - symbol`  (and `$ + symbol` as a fallback).
        Returns a dict such as: { "subtract": [ {"symbol":"$"}, {"symbol":"status_msg"} ] }
        """
        if not expr_container or not isinstance(expr_container[0], dict):
            return None

        actual = expr_container[0]
        # We only expect additive-style forms for typical equ uses.
        if "additiveExpression" not in actual:
            return None

        add_expr = actual["additiveExpression"]
        operands: List[Dict[str, str]] = []
        operators: List[str] = []

        for part in add_expr:
            # operand part (usually a castExpression)
            if isinstance(part, dict):
                if "castExpression" in part:
                    ce = part["castExpression"]
                    if ce and isinstance(ce[0], (list, tuple)):
                        # token form, e.g. ["$", "DOLLAR"]
                        tok = ce[0][0]
                        if tok == "$":
                            operands.append({"symbol": "$"})
                        else:
                            operands.append({"symbol": str(tok)})
                    elif ce and isinstance(ce[0], dict):
                        inner = ce[0]
                        if "name" in inner:
                            name = self.navigator.normalize_token(inner["name"])
                            operands.append({"symbol": name})
                        elif "register" in inner:
                            # unlikely in equ, but handle gracefully
                            operands.append({"symbol": self.navigator.normalize_token(inner["register"]).upper()})
            # operator token like ["-", "MINUS"]
            elif isinstance(part, list) and part:
                op = part[0]
                operators.append(op)

        # simple two-operand case: a - b  or  a + b
        if len(operators) == 1 and len(operands) >= 2:
            if operators[0] == "-":
                return {"subtract": operands[:2]}
            if operators[0] == "+":
                return {"add": operands[:2]}

        # Fallback: return additive list if available
        if operands:
            return {"add": operands}

        return None


    def _process_pseudo_value(self,
                              value_data: List[Any]) -> Optional[Dict[str, Any]]:
        """Interpret a single value inside a pseudo‑instruction."""
        if not value_data or not isinstance(value_data[0], dict):
            return None
        atom = value_data[0].get("atom", [{}])[0]

        if "integer" in atom:
            imm = self.registry.transform("integer", atom["integer"])
            return {"integer": {"type": imm.type, "value": imm.value}} if imm else None
        if "float_number" in atom:
            float_data = atom["float_number"][0]
            if float_data and len(float_data) >= 2:
                return {"float": {"type": float_data[1], "value": float_data[0]}}
        if "string" in atom:
            return {"string": atom["string"][0][0]} if atom["string"] else None
        if "expression" in atom:
            return self._process_expression(atom)
        return None

    def _process_expression(self, expr_container: Dict[str, Any]) -> Any:
        """Delegate expression handling to the specialised ``ExpressionVisitor``."""
        visitor = ExpressionVisitor(self.navigator)
        return visitor.process(expr_container)

    # -----------------------------------------------------------------
    # Small extraction helpers – keep the main flow readable
    # -----------------------------------------------------------------
    def _extract_section_name(self, directive_data: List[Any]) -> str:
        params = directive_data[1].get("section_params", [])
        if params and isinstance(params[0], dict):
            return self.navigator.normalize_token(params[0].get("name"))
        return ""

    def _extract_global_name(self, directive_data: List[Any]) -> str:
        params = directive_data[1].get("global_params", [])
        if params and isinstance(params[0], dict):
            return self.navigator.normalize_token(params[0].get("name"))
        return ""

    def _section_to_dict(self, section: Section) -> Dict[str, Any]:
        """Serialise a :class:`Section` dataclass to a plain dictionary."""
        return {
            "name": section.name,
            "lgroups": section.lgroups,
            "pseudo_instruct": section.pseudo_instruct
        }

    def _instruction_to_dict(self, instruction: Instruction) -> Dict[str, Any]:
        """Serialise an :class:`Instruction` dataclass, omitting empty fields."""
        result: Dict[str, Any] = {"opcode": instruction.opcode}
        if instruction.prefix:
            result["prefix"] = instruction.prefix
        if instruction.operands:
            result["operands"] = instruction.operands
        return result

# =============================================================================
# Expression visitor – parses the expression subtree used by operands
# =============================================================================
class ExpressionVisitor:
    """
    Handles the subset of expression grammar that appears inside operands.
    The visitor returns a lightweight representation (plain dicts / strings)
    rather than full ASTNode instances; this keeps the final JSON output
    compact while preserving all semantic information needed downstream.
    """
    def __init__(self, navigator: ParseTreeNavigator):
        self.navigator = navigator

    def process(self, expr_container: Dict[str, Any]) -> Any:
        if "expression" not in expr_container:
            return None
        expr_list = expr_container["expression"]
        if not expr_list or not isinstance(expr_list[0], dict):
            return None

        actual_expr = expr_list[0]
        if "castExpression" in actual_expr:
            return self._visit_cast(actual_expr["castExpression"])
        if "additiveExpression" in actual_expr:
            return self._visit_additive(actual_expr["additiveExpression"])
        if "multiplicativeExpression" in actual_expr:
            return self._visit_multiplicative(actual_expr["multiplicativeExpression"])
        return None

    # -----------------------------------------------------------------
    # Individual expression node handlers
    # -----------------------------------------------------------------
    def _visit_cast(self, cast_expr: List[Any]) -> Any:
        expr = cast_expr[0]
        if "name" in expr:
            return self.navigator.normalize_token(expr["name"])
        if "register" in expr:
            return {"register": self.navigator.normalize_token(expr["register"]).upper()}
        if "integer" in expr:
            val = expr["integer"][0]
            return {"integer": {"type": val[1], "value": val[0]}}
        if "unaryExpression" in expr:
            return self._visit_unary(expr["unaryExpression"])
        return None

    def _visit_unary(self, unary_expr: List[Any]) -> Any:
        """Handle a unary operation, e.g. ``-5``."""
        if len(unary_expr) == 2:
            op = self.navigator.normalize_token(unary_expr[0].get("unaryOperator", ""))
            operand = self.process({"expression": [unary_expr[1]]})

            if isinstance(operand, dict) and "integer" in operand:
                operand["integer"]["value"] = int(f"{op}{operand['integer']['value']}")
                return operand
            return {"unary_op": op, "unary_val": operand}
        return None

    def _visit_additive(self, add_expr: List[Any]) -> Any:
        """Collect operands of an additive expression (``+`` / ``-``)."""
        operands = []
        for comp in add_expr:
            if isinstance(comp, dict):
                if "multiplicativeExpression" in comp or "castExpression" in comp:
                    res = self.process({"expression": [comp]})
                    if isinstance(res, list):
                        res = {"multiplicative": res}
                    operands.append(res)
        return {"additive": operands} if len(operands) > 1 else (operands[0] if operands else None)

    def _visit_multiplicative(self, mul_expr: List[Any]) -> Any:
        """Collect operands of a multiplicative expression (``*`` / ``/``)."""
        operands = []
        for comp in mul_expr:
            if isinstance(comp, dict) and "castExpression" in comp:
                operands.append(self.process({"expression": [comp]}))
        return operands if len(operands) > 1 else (operands[0] if operands else None)

# =============================================================================
# Public API – entry point used by the command‑line wrapper
# =============================================================================
def transform(parse_tree: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function invoked by the CLI (or tests).  It builds an
    :class:`AsmTransformer` and runs ``visit_program`` on the ``program``
    key of the supplied parse tree.
    """
    transformer = AsmTransformer()
    program_data = parse_tree.get("program", [])
    return transformer.visit_program(program_data, {})

# =============================================================================
# CLI driver – reads a JSON representation of the parse tree from stdin
# =============================================================================
if __name__ == "__main__":
    try:
        parse_tree = json.load(sys.stdin)
        result = transform(parse_tree)
        json.dump(result, sys.stdout, indent=2)
    except json.JSONDecodeError:
        print("Error: Invalid JSON input.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Unexpected failures are reported with a concise message; the
        # exit code signals failure to the invoking process.
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
