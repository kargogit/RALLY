import json
from antlr4.tree.Tree import TerminalNodeImpl, ErrorNodeImpl
from antlr4 import *
from nasm_x86_64_lexer import nasm_x86_64_lexer
from nasm_x86_64_parser import nasm_x86_64_parser
from nasm_x86_64_parserListener import nasm_x86_64_parserListener
import sys

IGNORE_TOKENS     = {"EOL", "COLON"}
FLATTEN_WRAPPERS  = False

def leaf(tok, parser):
    return (tok.text, parser.symbolicNames[tok.type])

def tree_to_dict(node, parser):
    if isinstance(node, (TerminalNodeImpl, ErrorNodeImpl)):
        tok = node.getSymbol()
        if tok.channel != 0:
            return None
        if parser.symbolicNames[tok.type] in IGNORE_TOKENS:
            return None
        return leaf(tok, parser)

    rule_name = parser.ruleNames[node.getRuleIndex()]
    children = [tree_to_dict(ch, parser) for ch in node.getChildren()]
    children = [c for c in children if c is not None]

    if FLATTEN_WRAPPERS and len(children) == 1:
        return children[0]

    if(children):
        return {rule_name: children}
    else:
        return None

if len(sys.argv) < 2:
    print("Usage: python3 asmParser.py <input_asm_file>")
    sys.exit(1)

input_filename = sys.argv[1]

fileContent = ""
with open(input_filename, "r") as file:
    fileContent = file.read()

lexer = nasm_x86_64_lexer(InputStream(fileContent))
stream = CommonTokenStream(lexer)
parser = nasm_x86_64_parser(stream)

tree = parser.program()

#print(tree.toStringTree(recog=parser))
jsonStr = json.dumps(tree_to_dict(tree, parser), indent=2, ensure_ascii=False)
#print(type(tree_to_dict(tree, parser)))
print(jsonStr)
