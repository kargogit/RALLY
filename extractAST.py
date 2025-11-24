from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict
import json

# ============================================================================
# Node Classes - Better type safety and structure
# ============================================================================

@dataclass
class ASTNode:
    """Base class for all AST nodes"""
    pass

@dataclass
class Integer(ASTNode):
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
    type: str  # 'additive', 'multiplicative', 'unary', etc.
    operands: List[ASTNode] = field(default_factory=list)
    operator: Optional[str] = None

@dataclass
class Operand(ASTNode):
    register: Optional[str] = None
    size: Optional[str] = None
    expression: Optional[Expression] = None
    integer: Optional[Integer] = None
    name: Optional[str] = None

@dataclass
class Instruction(ASTNode):
    opcode: str
    operands: List[Operand] = field(default_factory=list)
    prefix: Optional[str] = None

@dataclass
class Label(ASTNode):
    name: str

@dataclass
class Section(ASTNode):
    name: str
    blocks: List[List[ASTNode]] = field(default_factory=list)
    pseudo_instruct: List[Dict[str, Any]] = field(default_factory=list)

# ============================================================================
# Helper Functions - Extraction utilities
# ============================================================================

class ParseTreeNavigator:
    """Utilities for navigating the parse tree"""

    @staticmethod
    def get_nested(data: Any, *keys: str, default=None) -> Any:
        """Safely get nested dictionary/list values"""
        current = data
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key, default)
            elif isinstance(current, list) and key.isdigit():
                idx = int(key)
                current = current[idx] if 0 <= idx < len(current) else default
            elif isinstance(current, list) and len(current) > 0:
                # Try to find key in first element if it's a dict
                if isinstance(current[0], dict):
                    current = current[0].get(key, default)
                else:
                    return default
            else:
                return default
            if current is None:
                return default
        return current

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
            for key in ("name", "register", "size", "opcode", "dx"):
                if key in token and token[key]:
                    return ParseTreeNavigator.normalize_token(token[key])

        return ""

# ============================================================================
# Visitor Pattern - Clean separation of traversal and transformation
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

        # Find the first key that matches a visit method
        for key in node.keys():
            method_name = f"visit_{key}"
            method = getattr(self, method_name, None)
            if method:
                return method(node[key], context)
            else:
                raise ValueError("Should be handled")

        return self.generic_visit(node, context)

    def generic_visit(self, node: Any, context: Dict[str, Any]) -> Any:
        """Called when no specific visit method exists"""
        return node

# ============================================================================
# Transformation Registry - For extensibility
# ============================================================================

class TransformationRegistry:
    """Registry pattern for registering transformation handlers"""

    def __init__(self):
        self._handlers: Dict[str, Callable] = {}

    def register(self, node_type: str):
        """Decorator to register a handler for a node type"""
        def decorator(func: Callable):
            self._handlers[node_type] = func
            return func
        return decorator

    def get_handler(self, node_type: str) -> Optional[Callable]:
        """Get handler for a node type"""
        return self._handlers.get(node_type)

    def transform(self, node_type: str, data: Any, context: Dict[str, Any] = None) -> Any:
        """Transform data using registered handler"""
        handler = self.get_handler(node_type)
        if handler:
            return handler(data, context or {})
        else:
            raise ValueError("Should be handled")

        return None

# ============================================================================
# Concrete Visitor Implementation
# ============================================================================

class AsmTransformer(ParseTreeVisitor):
    """Transforms ANTLR parse tree to simplified AST"""

    def __init__(self):
        super().__init__()
        self.registry = TransformationRegistry()
        self._setup_handlers()
        self.text_section = None

    def _setup_handlers(self):
        """Register all transformation handlers"""

        @self.registry.register("integer")
        def handle_integer(data, ctx):
            if isinstance(data, list) and len(data) > 0:
                int_data = data[0]
                if isinstance(int_data, (list, tuple)) and len(int_data) == 2:
                    return Integer(value=int_data[0], type=int_data[1])
                else:
                    raise ValueError("Should be handled")
            else:
                raise ValueError("Should be handled")

            return None

        @self.registry.register("register")
        def handle_register(data, ctx):
            name = self.navigator.normalize_token(data)
            return Register(name=name.upper())

        @self.registry.register("name")
        def handle_name(data, ctx):
            value = self.navigator.normalize_token(data)
            return Name(value=value)

    def visit_program(self, program_data: List[Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Transform top-level program"""
        output = []

        # Initialize sections
        text_section = Section(
            name = ".text"
        )
        global_store = {"globals": []}

        self.text_section = {"section": self._section_to_dict(text_section)}
        output.append(self.text_section)
        output.append(global_store)

        current_section = output[0]
        context["current_section"] = current_section

        for node in program_data:
            if not isinstance(node, dict):
                continue

            result = self.visit(node, context)
            if result:
                if isinstance(result, dict):
                    if "section" in result:
                        if result["section"]["name"] == ".text":
                            pass
                        else:
                            output.append(result)
                        context["current_section"] = result

                    elif "global" in result:
                        global_store["globals"].append(result["global"])

                    elif "block" in result:
                        context["current_section"]["section"]["blocks"].append(result)

                    elif "pseudo_instruct" in result:
                        context["current_section"]["section"]["pseudo_instruct"].append(result["pseudo_instruct"])

                    else:
                        raise ValueError("Should be handled")

                else:
                    raise ValueError("Should be handled")
            else:
                raise ValueError("Should be handled")

        return output

    def visit_line(self, line_data: List[Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a line (directive or instruction)"""
        if not line_data or not isinstance(line_data[0], dict):
            return None

        line_content = line_data[0]

        if "directive" in line_content:
            return self._process_directive(line_content["directive"], context)
        elif "pseudoinstruction" in line_content:
            return self._process_pseudoinstruction(line_content["pseudoinstruction"], context)
        else:
            raise ValueError("Should be handled")

        return None

    def visit_block(self, block_data: List[Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a block of instructions"""
        block_output = []

        for item in block_data:
            if not isinstance(item, dict):
                continue

            if "label" in item:
                label_node = self._process_label(item["label"])
                if label_node:
                    block_output.append(label_node)

            elif "non_terminator_line" in item or "terminator_line" in item:
                instr_node = self._process_instruction_line(item, context)
                if instr_node:
                    block_output.append(instr_node)
                else:
                    raise ValueError("Should be handled")

            else:
                raise ValueError("Should be handled")

        if block_output:
            return { "block": block_output }
        else:
            raise ValueError("Should be handled")


    def _process_directive(self, directive_data: List[Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process directive (section, global, etc.)"""
        if not directive_data or len(directive_data) < 2:
            return None

        directive_type = directive_data[0]
        directive_params = directive_data[1] if len(directive_data) > 1 else {}

        # Section directive
        if "section" in directive_type:
            name = self._extract_section_name(directive_data)
            if name == ".text":
                return self.text_section
            else:
                return {
                    "section": {
                        "name": name,
                        "blocks": [],
                        "pseudo_instruct": []
                    }
                }

        # Global directive
        elif "global" in directive_type:
            name = self._extract_global_name(directive_data)
            return {"global": name}

        else:
            raise ValueError("Should be handled")

        return None

    def _process_pseudoinstruction(self, pseudo_data: List[Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process pseudo-instruction"""
        pseudo_dict = {}
        pseudo_values = []

        for item in pseudo_data:
            if not isinstance(item, dict):
                continue

            if "name" in item:
                pseudo_dict["name"] = self.navigator.normalize_token(item["name"])
            elif "dx" in item:
                pseudo_dict["dx"] = self.navigator.normalize_token(item["dx"])
            elif "resx" in item:
                pseudo_dict["resx"] = self.navigator.normalize_token(item["resx"])
            elif "value" in item:
                value = self._process_pseudo_value(item["value"])
                if value:
                    pseudo_values.append(value)
            elif "integer" in item:
                int_node = self.registry.transform("integer", item["integer"])
                if int_node:
                    pseudo_dict["integer"] = {"type": int_node.type, "value": int_node.value}
                else:
                    raise ValueError("Should be handled")

        if pseudo_values:
            pseudo_dict["values"] = pseudo_values

        if pseudo_dict:
            return {"pseudo_instruct" : pseudo_dict}
        else:
            raise ValueError("Should be handled")

    def _process_pseudo_value(self, value_data: List[Any]) -> Optional[Dict[str, Any]]:
        """Process pseudo-instruction value"""
        if not value_data or not isinstance(value_data[0], dict):
            return None

        atom = value_data[0].get("atom", [{}])[0]

        if "integer" in atom:
            int_node = self.registry.transform("integer", atom["integer"])
            if int_node:
                return {"integer": {"type": int_node.type, "value": int_node.value}}
            else:
                raise ValueError("Should be handled")

        elif "float_number" in atom:
            float_data = atom["float_number"][0]
            if float_data and len(float_data) >= 2:
                return {"float": {"type": float_data[1], "value": float_data[0]}}
            else:
                raise ValueError("Should be handled")

        elif "string" in atom:
            str_data = atom["string"][0]
            if str_data:
                return {"string": str_data[0]}
            else:
                raise ValueError("Should be handled")

        elif "expression" in atom:
            return self._process_expression(atom)

        else:
            raise ValueError("Should be handled")

        return None

    def _process_label(self, label_data: List[Any]) -> Optional[Dict[str, Any]]:
        """Process label"""
        if not label_data or not isinstance(label_data[0], dict):
            return None

        name = self.navigator.normalize_token(label_data[0].get("name"))
        return {"label": name}

    def _process_instruction_line(self, line_item: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process instruction line"""
        line_node = line_item.get("non_terminator_line") or line_item.get("terminator_line")
        if not line_node or not isinstance(line_node[0], dict):
            return None

        instr_data = line_node[0].get("instruction") or line_node[0].get("terminator_instruction")
        if not instr_data:
            return None

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
                operand = self._process_operand(piece["operand"])
                instruction.operands.append(operand)
            else:
                raise ValueError("Should be handled")

        return {"instruction": self._instruction_to_dict(instruction)}

    def _process_operand(self, operand_data: List[Any]) -> Dict[str, Any]:
        """Process instruction operand"""
        operand = {}

        for item in operand_data:
            if not isinstance(item, dict):
                continue

            if "register" in item:
                operand["register"] = self.navigator.normalize_token(item["register"]).upper()
            elif "size" in item:
                operand["size"] = self.navigator.normalize_token(item["size"]).upper()
            elif "expression" in item:
                operand["expression"] = self._process_expression(item)
            elif "integer" in item:
                int_node = self.registry.transform("integer", item["integer"])
                if int_node:
                    operand["integer"] = {"value": int_node.value, "type": int_node.type}
                else:
                    raise ValueError("Should be handled")
            elif "name" in item:
                operand["name"] = self.navigator.normalize_token(item["name"])
            else:
                raise ValueError("Should be handled")

        return operand

    def _process_expression(self, expr_container: Dict[str, Any]) -> Any:
        """Process expression - simplified version"""
        # This would benefit from its own visitor class
        expr_visitor = ExpressionVisitor(self.navigator)
        return expr_visitor.process(expr_container)

    # Helper methods
    def _extract_section_name(self, directive_data: List[Any]) -> str:
        """Extract section name from directive"""
        if len(directive_data) > 1:
            params = directive_data[1].get("section_params", [])
            if params and isinstance(params[0], dict):
                return self.navigator.normalize_token(params[0].get("name"))
            else:
                raise ValueError("Should be handled")
        else:
            raise ValueError("Should be handled")
        return ""

    def _extract_global_name(self, directive_data: List[Any]) -> str:
        """Extract global name from directive"""
        if len(directive_data) > 1:
            params = directive_data[1].get("global_params", [])
            if params and isinstance(params[0], dict):
                return self.navigator.normalize_token(params[0].get("name"))
            else:
                raise ValueError("Should be handled")
        else:
            raise ValueError("Should be handled")
        return ""

    def _section_to_dict(self, section: Section) -> Dict[str, Any]:
        """Convert Section object to dict"""
        return {
            "name": section.name,
            "blocks": section.blocks,
            "pseudo_instruct": section.pseudo_instruct
        }

    def _instruction_to_dict(self, instruction: Instruction) -> Dict[str, Any]:
        """Convert Instruction object to dict"""
        result = {"opcode": instruction.opcode}
        if instruction.prefix:
            result["prefix"] = instruction.prefix
        if instruction.operands:
            result["operands"] = instruction.operands
        return result

# ============================================================================
# Specialized Expression Visitor
# ============================================================================

class ExpressionVisitor:
    """Dedicated visitor for expression processing"""

    def __init__(self, navigator: ParseTreeNavigator):
        self.navigator = navigator

    def process(self, expr_container: Dict[str, Any]) -> Any:
        """Process an expression container"""
        if "expression" not in expr_container:
            return None

        expr_list = expr_container["expression"]
        if not expr_list or not isinstance(expr_list[0], dict):
            return None

        actual_expr = expr_list[0]

        # Dispatch to specific handler
        if "castExpression" in actual_expr:
            return self._process_cast_expression(actual_expr["castExpression"])
        elif "additiveExpression" in actual_expr:
            return self._process_additive_expression(actual_expr["additiveExpression"])
        elif "multiplicativeExpression" in actual_expr:
            return self._process_multiplicative_expression(actual_expr["multiplicativeExpression"])
        else:
            raise ValueError("Should be handled")

        return None

    def _process_cast_expression(self, cast_expr: List[Any]) -> Any:
        """Process cast expression"""
        if not cast_expr or not isinstance(cast_expr[0], dict):
            return None

        expr = cast_expr[0]

        if "name" in expr:
            return self.navigator.normalize_token(expr["name"])

        elif "register" in expr:
            return {"register": self.navigator.normalize_token(expr["register"]).upper()}

        elif "integer" in expr:
            int_data = expr["integer"][0]
            if isinstance(int_data, (list, tuple)) and len(int_data) == 2:
                return {"integer": {"type": int_data[1], "value": int_data[0]}}
            else:
                raise ValueError("Should be handled")

        elif "unaryExpression" in expr:
            return self._process_unary_expression(expr["unaryExpression"])

        else:
            raise ValueError("Should be handled")

        return None

    def _process_unary_expression(self, unary_expr: List[Any]) -> Any:
        """Process unary expression"""
        if isinstance(unary_expr, list) and len(unary_expr) == 2:
            operator = self.navigator.normalize_token(unary_expr[0].get("unaryOperator", ""))
            operand_expr = unary_expr[1]

            # Recursively process the operand
            operand = self.process({"expression": [operand_expr]})

            # Special case: merge negative sign with integer
            if isinstance(operand, dict) and "integer" in operand:
                int_data = operand["integer"]
                if int_data["type"] == "DECIMAL_INTEGER":
                    return {
                        "integer": {
                            "type": int_data["type"],
                            "value": int(operator + str(int_data["value"]))
                        }
                    }
                else:
                    raise ValueError("Should be handled")
            else:
                raise ValueError("Should be handled")

            return {"unary_op": operator, "unary_val": operand}

        else:
            raise ValueError("Should be handled")

    def _process_additive_expression(self, add_expr: List[Any]) -> Any:
        """Process additive expression"""
        operands = []

        for component in add_expr:
            if isinstance(component, dict):
                if "multiplicativeExpression" in component or "castExpression" in component:
                    result = self.process({"expression": [component]})
                    if isinstance(result, list):
                        result = {"multiplicative": result}
                    elif isinstance(result, dict) and len(result) == 1:
                        pass
                    else:
                        raise ValueError("Should be handled")
                    operands.append(result)
                else:
                    raise ValueError("Should be handled")
            elif "PLUS" in component:
                continue
            else:
                raise ValueError("Should be handled")

        return {"additive": operands} if len(operands) > 1 else (operands[0] if operands else None)

    def _process_multiplicative_expression(self, mul_expr: List[Any]) -> Any:
        """Process multiplicative expression"""
        operands = []

        for component in mul_expr:
            if isinstance(component, dict) and "castExpression" in component:
                operands.append(self.process({"expression": [component]}))
            elif "MULTIPLICATION" in component:
                continue
            else:
                raise ValueError("Should be handled")

        return operands if len(operands) > 1 else (operands[0] if operands else None)

# ============================================================================
# Main Transformation Function
# ============================================================================

def transform(parse_tree: Dict[str, Any]) -> Dict[str, Any]:
    """Transform ANTLR parse tree to simplified AST"""
    transformer = AsmTransformer()
    program_data = parse_tree.get("program", [])

    context = {}
    transformed_program = transformer.visit_program(program_data, context)

    return {"program": transformed_program}

# ============================================================================
# Usage
# ============================================================================

if __name__ == "__main__":
    import sys
    import json
    import argparse

    parser = argparse.ArgumentParser(
        description="Transform an assembly language parse tree (JSON) into a simplified AST."
    )
    args = parser.parse_args()

    try:
        parse_tree = json.load(sys.stdin)

        result = transform(parse_tree)

        json.dump(result, sys.stdout, indent=2)

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input. {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
