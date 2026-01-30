#!/usr/bin/env python3
import json
import argparse
import sys
from antlr4.tree.Tree import TerminalNodeImpl, ErrorNodeImpl
from antlr4 import *
from nasm_x86_64_lexer import nasm_x86_64_lexer
from nasm_x86_64_parser import nasm_x86_64_parser
from nasm_x86_64_parserListener import nasm_x86_64_parserListener
from antlr4.ParserRuleContext import ParserRuleContext

# tokens to ignore when turning tree into dict (unchanged)
IGNORE_TOKENS     = {"EOL", "COLON"}
FLATTEN_WRAPPERS  = False

# will be set from CLI
SLOC_ENABLED = False

def leaf(tok, parser):
    return (tok.text, parser.symbolicNames[tok.type])

def tree_to_dict(node, parser):
    """
    Convert an ANTLR parse-tree node to a dict representation.
    When SLOC_ENABLED is True, adds an '_loc' entry for ParserRuleContext nodes
    that have start/stop token information.
    """
    # Terminal / error nodes
    if isinstance(node, (TerminalNodeImpl, ErrorNodeImpl)):
        tok = node.getSymbol()
        # token channels other than 0 are usually hidden (whitespace/comments)
        if getattr(tok, 'channel', 0) != 0:
            return None
        token_name = parser.symbolicNames[tok.type] if tok.type < len(parser.symbolicNames) else None
        if token_name in IGNORE_TOKENS:
            return None
        return leaf(tok, parser)

    # Rule nodes
    rule_name = parser.ruleNames[node.getRuleIndex()]
    children = [tree_to_dict(ch, parser) for ch in node.getChildren()]
    children = [c for c in children if c is not None]

    if not children:
        result = None
    else:
        result = {rule_name: children}

    # Add source-location info only if enabled and the node has token bounds
    if SLOC_ENABLED and result is not None and isinstance(node, ParserRuleContext) and getattr(node, "start", None) is not None:
        start_tok = node.start
        stop_tok = node.stop if getattr(node, "stop", None) is not None else start_tok

        # calculate end column carefully (stop_tok.text may be None or empty)
        stop_text = stop_tok.text if getattr(stop_tok, 'text', None) is not None else ""
        end_col = (stop_tok.column + len(stop_text) - 1) if stop_text else stop_tok.column

        result['_loc'] = {
            'start': {'line': start_tok.line, 'column': start_tok.column},
            'end':   {'line': stop_tok.line,   'column': end_col}
        }

    if FLATTEN_WRAPPERS and isinstance(children, list) and len(children) == 1:
        return children[0]

    return result

def main(argv):
    global SLOC_ENABLED

    cli = argparse.ArgumentParser(
        description="Parse NASM x86-64 assembly and dump a JSON parse tree. "
                    "Use -sloc to include source-location metadata (_loc keys)."
    )
    cli.add_argument("input_file", help="Assembly input file")
    cli.add_argument("-sloc", "--sloc", action="store_true",
                     help="collect and include source-location info in the parse tree")
    args = cli.parse_args(argv[1:])

    SLOC_ENABLED = args.sloc

    with open(args.input_file, "r", encoding="utf-8") as f:
        fileContent = f.read()

    lexer = nasm_x86_64_lexer(InputStream(fileContent))
    stream = CommonTokenStream(lexer)
    parser = nasm_x86_64_parser(stream)

    tree = parser.program()

    # convert and pretty-print JSON
    parsed = tree_to_dict(tree, parser)
    jsonStr = json.dumps(parsed, indent=2, ensure_ascii=False)
    print(jsonStr)

if __name__ == "__main__":
    main(sys.argv)
