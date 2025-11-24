from typing import Any, Dict, List
import json

def normalize_token(tok: Any) -> str:
    """
    Extract the string value from a token node.
    Accepts:
      - ('name', 'VALUE') -> 'VALUE'
      - ['name', ['VALUE']] -> 'VALUE'
      - {'name': [...]} -> 'VALUE'
    """
    if (isinstance(tok, tuple) or isinstance(tok, list)) and len(tok) == 2 and isinstance(tok[0], str):
        return str(tok[0])
    if isinstance(tok, list):
        if tok and isinstance(tok[0], str) and isinstance(tok[1], list):
            return str(tok[1][0])
        if len(tok) == 1:
            return normalize_token(tok[0])
    if isinstance(tok, dict):
        for k in ("name", "register", "size", "opcode", "dx"):
            if k in tok and tok[k]:
                return normalize_token(tok[k])
    if isinstance(tok, str):
        return tok
    raise ValueError(f"Cannot normalize token: {tok}")

def processExpression(expression: Dict[str, Any]) -> Any:
    actualExpress = expression["expression"][0]
    if "castExpression" in actualExpress:
        castExpress = actualExpress["castExpression"][0]
        if "name" in castExpress:
            return normalize_token(castExpress["name"]) if castExpress["name"] else ""
        elif "unaryExpression" in castExpress:
            unaryExpress = castExpress["unaryExpression"]
            express_data = {}
            if isinstance(unaryExpress, list) and len(unaryExpress) == 2:
                if "unaryOperator" in unaryExpress[0]:
                    unaryOp = unaryExpress[0]["unaryOperator"]
                    express_data["unary_op"] = normalize_token(unaryOp) if unaryOp else ""
                else:
                    raise ValueError("Should be handled")
                if "castExpression" in unaryExpress[1]:
                    internalExpress = processExpression({"expression": [unaryExpress[1]]})
                    express_data["unary_val"] = internalExpress
                else:
                    raise ValueError("Should be handled")
            else:
                raise ValueError("Should be handled")
            if express_data.get("unary_op") and express_data.get("unary_val"):
                if "integer" in express_data["unary_val"]:
                    int_data = express_data["unary_val"]["integer"]
                    if int_data["type"] == "DECIMAL_INTEGER":
                        return {"integer": { "type": int_data["type"], "value": int(express_data["unary_op"] + express_data["unary_val"]["integer"]["value"]) } }
                    else:
                        raise ValueError("Should be handled")
                else:
                    raise ValueError("Should be handled")
            else:
                raise ValueError("Should be handled")
        elif "register" in castExpress:
            express_data = {}
            express_data["register"] = normalize_token(castExpress["register"]).upper() if castExpress["register"] else ""
            return express_data
        elif "integer" in castExpress and len(castExpress["integer"][0]) == 2:
            int_express = castExpress["integer"][0]
            int_val = int_express[0]
            int_type = int_express[1]
            return {"integer": { "type": int_type, "value": int_val } }
        else:
            raise ValueError("Should be handled")
    elif "additiveExpression" in actualExpress:
        addExpress = actualExpress["additiveExpression"]
        if ['+', "PLUS"] in addExpress:
            plusCount = addExpress.count(['+', "PLUS"])
            if len(addExpress) == ( (plusCount + 1) + plusCount ):
                thingsToAdd = []
                for expressComponent in addExpress:
                    if "multiplicativeExpression" in expressComponent:
                        mulExpress = processExpression({"expression": [expressComponent]})
                        if isinstance(mulExpress, list):
                            mulExpress = {"multiplicative": mulExpress}
                        thingsToAdd.append(mulExpress)
                    elif "PLUS" in expressComponent:
                        continue
                    elif "castExpression" in expressComponent:
                        thingsToAdd.append(processExpression( { "expression": [expressComponent] } ))
                    else:
                        raise ValueError("Should be handled")
                if len(thingsToAdd) == 1:
                    return thingsToAdd[0]
                elif len(thingsToAdd) > 1:
                    return {"additive": thingsToAdd}
                else:
                    raise ValueError("Should be handled")
            else:
                raise ValueError("Should be handled")
        else:
            raise ValueError("Should be handled")
    elif "multiplicativeExpression" in actualExpress:
        mulExpress = actualExpress["multiplicativeExpression"]
        mulAST = []
        for expressComponent in mulExpress:
            mulCount = mulExpress.count(['*', "MULTIPLICATION"])
            if len(mulExpress) == ( (mulCount + 1) + mulCount ):
                if "castExpression" in expressComponent:
                    mulAST.append( processExpression( { "expression": [expressComponent] } ) )
                elif "MULTIPLICATION" in expressComponent:
                    continue
                else:
                    raise ValueError("Should be handled")
            else:
                raise ValueError("Should be handled")
        if len(mulAST) == 1:
            return mulAST[0]
        else:
            return mulAST
    else:
        raise ValueError("Should be handled")


def extract_operand(operand_list: List[Any]) -> Dict[str, Any]:
    operand_dict = {}
    for item in operand_list:
        if "register" in item:
            register = item["register"]
            register = normalize_token(register).upper() if register else ""
            operand_dict["register"] = register
        elif "size" in item:
            size = item["size"]
            size = normalize_token(size).upper() if size else ""
            operand_dict["size"] = size
        elif "expression" in item:
            operand_dict["expression"] = processExpression(item)
        elif "integer" in item:
            if isinstance(item["integer"], list) and len(item["integer"][0]) == 2:
                integer = item["integer"][0]
                operand_dict["integer"] = { "value": integer[0], "type": integer[1] }
            else:
                raise ValueError("Should be handled")
        elif "name" in item:
            if isinstance(item["name"], list) and len(item["name"][0]) == 2:
                name = item["name"][0]
                name = normalize_token(name) if name else ""
                operand_dict["name"] = name
        elif "[" in item or "]" in item:
            continue
        else:
            raise ValueError("Should be handled")
    return operand_dict

def transform_program(program_node: List[Any]) -> List[Dict[str, Any]]:
    """
    Transform the top-level 'program' list into the desired structure.
    """
    out = []
    textSection = {
                        "directive": {
                            "name": "section",
                            "args": [{"type": "NAME", "value": ".text"}],
                            "blocks": []
                        }
                    }
    globalStore = {"globals": []}
    out.append(textSection)
    out.append(globalStore)
    currentSection = textSection

    for node in program_node:
        if not isinstance(node, dict):
            continue

        # line -> directive
        if "line" in node:
            if "directive" in node["line"][0]:
                d = node["line"][0]["directive"]
                # section
                if "section" in d[0]:
                    params = d[1].get("section_params")
                    name_node = None
                    if params:
                        name_node = params[0].get("name")
                    elif "section_params" in d:
                        name_node = d["section_params"][0].get("name")
                    name = normalize_token(name_node) if name_node else ""
                    if name == ".text":
                        currentSection = textSection
                        continue
                    else:
                        newSection = {
                            "directive": {
                                "name": "section",
                                "args": [{"type": "NAME", "value": name}],
                                "blocks": [],
                                "pseudo_instruct": []
                            }
                        }
                        out.append(newSection)
                        currentSection = newSection
                        continue

                #global
                elif "global" in d[0]:
                    params = d[1].get("global_params")
                    name_node = None
                    if params:
                        name_node = params[0].get("name")
                    elif "global_params" in d:
                        name_node = d["global_params"][0].get("name")
                    name = normalize_token(name_node) if name_node else ""
                    globalStore["globals"].append( {"type": "NAME", "value": name} )
                    continue
            elif "pseudoinstruction" in node["line"][0]:
                pseudo = node["line"][0]["pseudoinstruction"]
                pseudo_data = {}
                pseudo_values = []
                for item in pseudo:
                    if "name" in item:
                        name = normalize_token(item["name"]) if item["name"] else ""
                        pseudo_data["name"] = name
                    elif "dx" in item:
                        dx = normalize_token(item["dx"]) if item["dx"] else ""
                        pseudo_data["dx"] = dx
                    elif "value" in item:
                        value_node = item["value"][0]["atom"][0]
                        if "integer" in value_node:
                            int_node = value_node["integer"][0]
                            if int_node:
                                int_val = int_node[0]
                                int_type = int_node[1]
                                pseudo_values.append({"integer": { "type": int_type, "value": int_val } })
                            continue

                        elif "float_number" in value_node:
                            float_node = value_node["float_number"][0]
                            if float_node:
                                float_val = float_node[0]
                                float_type = float_node[1]
                                pseudo_values.append({"float": { "type": float_type, "value": float_val } })
                            continue

                        elif "string" in value_node:
                            str_node = value_node["string"][0]
                            if str_node:
                                str_val = str_node[0]
                                pseudo_values.append({"string": str_val})
                            continue

                        elif "expression" in value_node:
                            pseudo_values.append(processExpression(value_node))

                        else:
                            raise ValueError("Should be handled")
                    elif "resx" in item:
                        resx = normalize_token(item["resx"]) if item["resx"] else ""
                        pseudo_data["resx"] = resx
                    elif "integer" in item:
                        int_node = item["integer"][0]
                        if int_node:
                            int_val = int_node[0]
                            int_type = int_node[1]
                            pseudo_data["integer"] = { "type": int_type, "value": int_val }
                    elif "COMMA" in item:
                        continue
                    else:
                        raise ValueError("Should be handled")
                if pseudo_values:
                    pseudo_data["values"] = pseudo_values
                currentSection["directive"]["pseudo_instruct"].append(pseudo_data)
            else:
                raise ValueError("Should be handled")

        # block
        elif "block" in node:
            block_seq = node["block"]
            block_out = []
            for item in block_seq:
                if not isinstance(item, dict):
                    continue
                # label
                if "label" in item:
                    name_node = item["label"][0].get("name")
                    name = normalize_token(name_node) if name_node else ""
                    block_out.append({"label": {"type": "NAME", "value": name}})
                    continue

                # non_terminator_line -> instruction
                if "non_terminator_line" in item or "terminator_line" in item:
                    line_node = item.get("non_terminator_line") or item.get("terminator_line")
                    instr = line_node[0].get("instruction") or line_node[0].get("terminator_instruction")
                    if instr:
                        prefix = None
                        opcode = None
                        operands = []
                        for instructPiece in instr:
                            if isinstance(instructPiece, dict):
                                if "lock_prefix" in instructPiece:
                                    prefix = "LOCK"
                                elif "opcode" in instructPiece:
                                    opcode = instructPiece["opcode"]
                                    opcode = normalize_token(opcode).upper() if opcode else ""
                                elif "terminator_opcode" in instructPiece:
                                    opcode = instructPiece["terminator_opcode"]
                                    opcode = normalize_token(opcode).upper() if opcode else ""
                                elif "operand" in instructPiece:
                                    operand = extract_operand(instructPiece["operand"])
                                    operands.append(operand)
                                else:
                                    #print(instructPiece)
                                    raise ValueError("Should be handled")
                            elif "COMMA" in instructPiece:
                                continue
                            else:
                                raise ValueError("Should be handled")
                        astInstr = {}
                        if prefix:
                            astInstr["prefix"] = "LOCK"
                        if opcode:
                            astInstr["opcode"] = opcode
                        if operands:
                            astInstr["operands"] = operands
                        block_out.append({"instruction": astInstr})
                        continue

            currentSection["directive"]["blocks"].append(block_out)
            continue

        else:
            raise ValueError("Should be handled")

    return out

def transform(ast: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform the full AST into the target structure.
    """
    program = ast.get("program", [])
    transformed_program = transform_program(program)
    return {"program": transformed_program}


# Example usage with the provided AST
if __name__ == "__main__":
    fileName = "testParseTree"
    with open(fileName, 'r', encoding = "utf-8") as fPtr:
        fileContent = fPtr.read()
    parseTree = json.loads(fileContent)

    result = transform(parseTree)
    import json
    print(json.dumps(result, indent=2))
