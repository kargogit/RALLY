from typing import Any, Dict, List, Optional, Union, Callable, Protocol
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict
import json
import sys

# ============================================================================
# ASTVisitor Protocol
# ============================================================================

class ASTVisitor(Protocol):
    """Protocol defining explicit visit methods for AST nodes"""
    def visit_program(self, node: Any, context: Any) -> Any: ...
    def visit_section(self, node: Any, context: Any) -> Any: ...
    def visit_block(self, node: Any, context: Any) -> Any: ...
    def visit_instruction(self, node: Any, context: Any) -> Any: ...
    def visit_operand(self, node: Any, context: Any) -> Any: ...
    def visit_label(self, node: Any, context: Any) -> Any: ...
    def generic_visit(self, node: Any, context: Any) -> Any: ...

# ============================================================================
# AST Node Definitions (Hierarchical & Machine-Independent)
# ============================================================================

@dataclass
class ASTNode:
    """Base class for all AST nodes"""
    pass

@dataclass
class Program(ASTNode):
    """Root program node"""
    sections: List['Section'] = field(default_factory=list)

@dataclass
class Section(ASTNode):
    """Encapsulates code sections (e.g., .text, .data)"""
    name: str
    blocks: List['BasicBlock'] = field(default_factory=list)
    pseudo_instruct: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class BasicBlock(ASTNode):
    """A logical block of instructions and labels"""
    instructions: List[Union['Instruction', 'Label']] = field(default_factory=list)

@dataclass
class Instruction(ASTNode):
    """Captures opcodes and operands"""
    opcode: str
    operands: List['Operand'] = field(default_factory=list)
    prefix: Optional[str] = None

@dataclass
class Label(ASTNode):
    """Represents symbolic identifiers"""
    name: str

@dataclass
class Operand(ASTNode):
    """Generic operand container"""
    register: Optional[str] = None
    memory: Optional['Memory'] = None
    immediate: Optional['Immediate'] = None
    expression: Optional['Expression'] = None
    size: Optional[str] = None
    name: Optional[str] = None

@dataclass
class Memory(ASTNode):
    """Specialized Memory operand (including RIP-relative)"""
    base: Optional[str] = None
    index: Optional[str] = None
    scale: Optional[int] = None
    displacement: Optional[Union[int, str]] = None

@dataclass
class Immediate(ASTNode):
    """Specialized Immediate operand"""
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

# ============================================================================
# Helper Functions
# ============================================================================

class ParseTreeNavigator:
    """Utilities for navigating the parse tree"""

    @staticmethod
    def normalize_token(token: Any) -> str:
        """Extract string value from various token formats"""
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
            for key in ("name", "register", "size", "opcode", "dx", "terminator_opcode"):
                if key in token and token[key]:
                    return ParseTreeNavigator.normalize_token(token[key])
        return ""

# ============================================================================
# Visitor Pattern Base
# ============================================================================

class ParseTreeVisitor(ABC):
    """Base visitor for parse tree traversal"""

    def __init__(self):
        self.navigator = ParseTreeNavigator()

    def visit(self, node: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """Dispatch to appropriate visit method based on node type"""
        if not isinstance(node, dict):
            return None
        context = context or {}

        # First matching key dispatch strategy
        for key in node.keys():
            method_name = f"visit_{key}"
            if hasattr(self, method_name):
                return getattr(self, method_name)(node[key], context)

        return self.generic_visit(node, context)

    def generic_visit(self, node: Any, context: Dict[str, Any]) -> Any:
        return node

# ============================================================================
# Transformation Registry
# ============================================================================

class TransformationRegistry:
    """Registry for granular node transformations"""
    def __init__(self):
        self._handlers: Dict[str, Callable] = {}

    def register(self, node_type: str):
        def decorator(func: Callable):
            self._handlers[node_type] = func
            return func
        return decorator

    def transform(self, node_type: str, data: Any) -> Any:
        handler = self._handlers.get(node_type)
        return handler(data, {}) if handler else None

# ============================================================================
# Concrete Visitor Implementation
# ============================================================================

class AsmTransformer(ParseTreeVisitor):
    """
    Concrete visitor that traverses the ANTLR parse tree and builds
    a hierarchical, machine-independent AST.
    """

    def __init__(self):
        super().__init__()
        self.registry = TransformationRegistry()
        self._setup_handlers()
        # Initialize default .text section
        self.text_section = Section(name=".text")

    def _setup_handlers(self):
        @self.registry.register("integer")
        def handle_integer(data, ctx):
            if isinstance(data, list) and len(data) > 0:
                int_data = data[0]
                if isinstance(int_data, (list, tuple)) and len(int_data) == 2:
                    return Immediate(value=int_data[0], type=int_data[1])
            return None

    # ============================================================================
    # Specialized Visitors (Concrete Delegation)
    # ============================================================================

    def visit_program(self, program_data: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Entry point: Transforms Program -> Sections"""
        output = []

        # Default section structure
        text_section_dict = self._section_to_dict(self.text_section)
        global_store = {"globals": []}

        # We keep a reference to the current dict we are appending to
        output.append({"section": text_section_dict})
        output.append(global_store)

        context["current_section_dict"] = output[0]

        for node in program_data:
            if not isinstance(node, dict):
                continue

            # Dispatch to specific line/block handlers
            result = self.visit(node, context)

            if result:
                if "section" in result:
                    # New section found
                    if result["section"]["name"] == ".text":
                        pass # Already initialized
                    else:
                        output.append(result)
                    context["current_section_dict"] = result
                elif "global" in result:
                    global_store["globals"].append(result["global"])
                elif "block" in result:
                    context["current_section_dict"]["section"]["blocks"].append(result)
                elif "pseudo_instruct" in result:
                    context["current_section_dict"]["section"]["pseudo_instruct"].append(result["pseudo_instruct"])

        return {"program": output}

    def visit_line(self, line_data: List[Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a line (Delegates to Directive or Instruction)"""
        if not line_data or not isinstance(line_data[0], dict):
            return None

        line_content = line_data[0]
        if "directive" in line_content:
            return self._process_directive(line_content["directive"], context)
        elif "pseudoinstruction" in line_content:
            return self._process_pseudoinstruction(line_content["pseudoinstruction"], context)
        return None

    def visit_block(self, block_data: List[Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a Basic Block of instructions/labels"""
        block_nodes = []

        for item in block_data:
            if not isinstance(item, dict):
                continue

            if "label" in item:
                label_node = self.visit_label(item["label"], context)
                if label_node:
                    block_nodes.append(label_node)
            elif "non_terminator_line" in item or "terminator_line" in item:
                # Extract instruction data from line wrapper
                line_node = item.get("non_terminator_line") or item.get("terminator_line")
                instr_data = line_node[0].get("instruction") or line_node[0].get("terminator_instruction")
                if instr_data:
                    instr_node = self.visit_instruction(instr_data, context)
                    if instr_node:
                        block_nodes.append(instr_node)

        return {"block": block_nodes} if block_nodes else None

    def visit_label(self, label_data: List[Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Explicit visitor for Label nodes"""
        if not label_data or not isinstance(label_data[0], dict):
            return None
        name = self.navigator.normalize_token(label_data[0].get("name"))
        return {"label": name}

    def visit_instruction(self, instr_data: List[Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Explicit visitor for Instruction nodes"""
        instruction = Instruction(opcode="")

        for piece in instr_data:
            if not isinstance(piece, dict):
                continue

            if "lock_prefix" in piece:
                instruction.prefix = "LOCK"
            elif "opcode" in piece or "terminator_opcode" in piece:
                opcode_key = "opcode" if "opcode" in piece else "terminator_opcode"
                instruction.opcode = self.navigator.normalize_token(piece[opcode_key]).upper()
            elif "operand" in piece:
                operand = self.visit_operand(piece["operand"], context)
                instruction.operands.append(operand)

        return {"instruction": self._instruction_to_dict(instruction)}

    def visit_operand(self, operand_data: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Explicit visitor for Operand nodes"""
        operand_dict = {}

        # 1. Detect RIP-Relative addressing for position independence
        rip_memory = self._detect_rip_relative(operand_data)
        if rip_memory:
            operand_dict["memory"] = {
                "base": rip_memory.base,
                "displacement": rip_memory.displacement
            }
            return operand_dict

        # 2. Standard processing
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

    # ============================================================================
    # RIP-Relative & Semantic Extraction Logic
    # ============================================================================

    def _detect_rip_relative(self, operand_data: Any) -> Optional[Memory]:
        """Detects RIP-relative addressing in operand expressions."""
        if not operand_data:
            return None

        # Simple heuristic: check string representation for rip/rel keyword
        op_str = str(operand_data).lower()
        if 'rip' in op_str or 'rel' in op_str:
            return Memory(
                base="RIP",
                displacement=self._extract_rip_displacement(operand_data)
            )
        return None

    def _extract_rip_displacement(self, operand_data: Any) -> Optional[Union[int, str]]:
        """Extracts the symbolic displacement (e.g., 'stderr' from [rel stderr])"""
        if isinstance(operand_data, list):
            for item in operand_data:
                if isinstance(item, dict) and 'expression' in item:
                    expr = item['expression']
                    # Dive into additive expression to find the label
                    if isinstance(expr, list) and len(expr) > 0:
                        return self._extract_from_expression(expr[0])
        return None

    def _extract_from_expression(self, expr_data: Any) -> Optional[Union[int, str]]:
        """Recursively looks for the displacement identifier"""
        if not isinstance(expr_data, dict):
            return None

        # Check for additive expression structure
        if 'additiveExpression' in expr_data:
            add_expr = expr_data['additiveExpression']
            # Usually the second part of [rel + Label]
            if len(add_expr) > 1:
                for part in add_expr:
                    if isinstance(part, dict):
                        # Try to find name/label
                        res = self._extract_from_expression(part)
                        if res: return res

        # Check basic atoms
        if 'multiplicativeExpression' in expr_data:
            return self._extract_from_expression(expr_data['multiplicativeExpression'][0])
        if 'castExpression' in expr_data:
             return self._extract_from_expression(expr_data['castExpression'][0])
        if 'unaryExpression' in expr_data:
             return self._extract_from_expression(expr_data['unaryExpression'][1])

        # Base cases
        if 'name' in expr_data:
            return self.navigator.normalize_token(expr_data['name'])
        if 'integer' in expr_data:
             val = self.registry.transform("integer", expr_data['integer'])
             return val.value if val else None

        return None

    # ============================================================================
    # Helper Processors (Directives, Pseudo, Expressions)
    # ============================================================================

    def _process_directive(self, directive_data: List[Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not directive_data or len(directive_data) < 2:
            return None

        directive_type = directive_data[0]

        if "section" in directive_type:
            name = self._extract_section_name(directive_data)
            if name == ".text":
                return {"section": self._section_to_dict(self.text_section)}
            else:
                # Create new section node
                new_sect = Section(name=name)
                return {"section": self._section_to_dict(new_sect)}

        elif "global" in directive_type:
            name = self._extract_global_name(directive_data)
            return {"global": name}

        return None

    def _process_pseudoinstruction(self, pseudo_data: List[Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pseudo_dict = {}
        pseudo_values = []

        for item in pseudo_data:
            if not isinstance(item, dict): continue

            if "name" in item:
                pseudo_dict["name"] = self.navigator.normalize_token(item["name"])
            elif "dx" in item:
                pseudo_dict["dx"] = self.navigator.normalize_token(item["dx"])
            elif "resx" in item:
                pseudo_dict["resx"] = self.navigator.normalize_token(item["resx"])
            elif "value" in item:
                val = self._process_pseudo_value(item["value"])
                if val: pseudo_values.append(val)
            elif "integer" in item:
                imm = self.registry.transform("integer", item["integer"])
                if imm: pseudo_dict["integer"] = {"type": imm.type, "value": imm.value}

        if pseudo_values: pseudo_dict["values"] = pseudo_values
        return {"pseudo_instruct": pseudo_dict} if pseudo_dict else None

    def _process_pseudo_value(self, value_data: List[Any]) -> Optional[Dict[str, Any]]:
        if not value_data or not isinstance(value_data[0], dict): return None
        atom = value_data[0].get("atom", [{}])[0]

        if "integer" in atom:
            imm = self.registry.transform("integer", atom["integer"])
            return {"integer": {"type": imm.type, "value": imm.value}} if imm else None
        elif "float_number" in atom:
            float_data = atom["float_number"][0]
            if float_data and len(float_data) >= 2:
                return {"float": {"type": float_data[1], "value": float_data[0]}}
        elif "string" in atom:
            return {"string": atom["string"][0][0]} if atom["string"] else None
        elif "expression" in atom:
            return self._process_expression(atom)
        return None

    def _process_expression(self, expr_container: Dict[str, Any]) -> Any:
        visitor = ExpressionVisitor(self.navigator)
        return visitor.process(expr_container)

    # Data Extraction Helpers
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
        return {
            "name": section.name,
            "blocks": section.blocks,
            "pseudo_instruct": section.pseudo_instruct
        }

    def _instruction_to_dict(self, instruction: Instruction) -> Dict[str, Any]:
        result = {"opcode": instruction.opcode}
        if instruction.prefix: result["prefix"] = instruction.prefix
        if instruction.operands: result["operands"] = instruction.operands
        return result

# ============================================================================
# Expression Visitor (Complex Operand Logic)
# ============================================================================

class ExpressionVisitor:
    def __init__(self, navigator: ParseTreeNavigator):
        self.navigator = navigator

    def process(self, expr_container: Dict[str, Any]) -> Any:
        if "expression" not in expr_container: return None
        expr_list = expr_container["expression"]
        if not expr_list or not isinstance(expr_list[0], dict): return None

        actual_expr = expr_list[0]
        if "castExpression" in actual_expr:
            return self._visit_cast(actual_expr["castExpression"])
        elif "additiveExpression" in actual_expr:
            return self._visit_additive(actual_expr["additiveExpression"])
        elif "multiplicativeExpression" in actual_expr:
            return self._visit_multiplicative(actual_expr["multiplicativeExpression"])
        return None

    def _visit_cast(self, cast_expr: List[Any]) -> Any:
        expr = cast_expr[0]
        if "name" in expr:
            return self.navigator.normalize_token(expr["name"])
        elif "register" in expr:
            return {"register": self.navigator.normalize_token(expr["register"]).upper()}
        elif "integer" in expr:
            val = expr["integer"][0]
            return {"integer": {"type": val[1], "value": val[0]}}
        elif "unaryExpression" in expr:
            return self._visit_unary(expr["unaryExpression"])
        return None

    def _visit_unary(self, unary_expr: List[Any]) -> Any:
        if len(unary_expr) == 2:
            op = self.navigator.normalize_token(unary_expr[0].get("unaryOperator", ""))
            operand = self.process({"expression": [unary_expr[1]]})

            if isinstance(operand, dict) and "integer" in operand:
                operand["integer"]["value"] = int(f"{op}{operand['integer']['value']}")
                return operand
            return {"unary_op": op, "unary_val": operand}
        return None

    def _visit_additive(self, add_expr: List[Any]) -> Any:
        operands = []
        for comp in add_expr:
            if isinstance(comp, dict):
                if "multiplicativeExpression" in comp or "castExpression" in comp:
                    res = self.process({"expression": [comp]})
                    if isinstance(res, list): res = {"multiplicative": res}
                    operands.append(res)
        return {"additive": operands} if len(operands) > 1 else (operands[0] if operands else None)

    def _visit_multiplicative(self, mul_expr: List[Any]) -> Any:
        operands = []
        for comp in mul_expr:
            if isinstance(comp, dict) and "castExpression" in comp:
                operands.append(self.process({"expression": [comp]}))
        return operands if len(operands) > 1 else (operands[0] if operands else None)

# ============================================================================
# Main Entry Point
# ============================================================================

def transform(parse_tree: Dict[str, Any]) -> Dict[str, Any]:
    transformer = AsmTransformer()
    program_data = parse_tree.get("program", [])
    return transformer.visit_program(program_data, {})

if __name__ == "__main__":
    try:
        parse_tree = json.load(sys.stdin)
        result = transform(parse_tree)
        json.dump(result, sys.stdout, indent=2)
    except json.JSONDecodeError:
        print("Error: Invalid JSON input.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
