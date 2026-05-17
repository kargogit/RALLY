"""
interprocAnalysis.py
Step 9: Interprocedural Propagation and Signature Finalization (Revised).
Performs targeted interprocedural refinement of widths, signedness, pointer
distinctions and scalar FP types. Finalizes dual signatures:
  - 'lifted_signature' (ret_type (%State*)) for EVERY function (void if noreturn)
  - 'external_abi_signature' for boundary functions only (refined from Step 7
    while preserving standard structure).
Includes noreturn propagation + CFG pruning of fall-through edges.
Inputs/Outputs: Enriched AST (JSON via stdin → Protobuf via stdout/file).
"""
import sys
import json
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict
from google.protobuf import text_format
from google.protobuf.json_format import ParseDict, MessageToJson
import ast_pb2 as pb
from astNodes import (
    legacy_program_dict_to_ast,
    ast_to_legacy_section_dict,
    Program,
    Function,
    BasicBlock,
    Instruction,
    LiftedFunctionSignature
)
# ---------------------------------------------------------------------------
# Protobuf Serialization Helpers
# ---------------------------------------------------------------------------
def normalize_section(sec_dict):
    """Wrap naked list forms to fulfill Proto message schemas before feeding to ParseDict"""
    if 'children' in sec_dict:
        for child in sec_dict['children']:
            if 'lgroup' in child and isinstance(child['lgroup'], list):
                child['lgroup'] = {'items': child['lgroup']}
            if 'dgroup' in child and isinstance(child['dgroup'], list):
                child['dgroup'] = {'items': child['dgroup']}
    return sec_dict
def ast_to_proto(ast_obj: Program, original_dict: dict) -> pb.Program:
    """Serializes the optimized AST directly into a Protobuf message."""
    normalized = {
        "sections": [
            normalize_section(ast_to_legacy_section_dict(s, include_instr_locations=False, include_enhancements=True))
            for s in ast_obj.sections
        ],
        "globals": ast_obj.globals,
        "symbol_table": original_dict.get("symbol_table", {}),
        "id_maps": original_dict.get("id_maps", {})
    }
    proto = pb.Program()
    ParseDict(normalized, proto, ignore_unknown_fields=True)
    return proto
# ---------------------------------------------------------------------------
# Type System (improved for FP, ptr retention, void)
# ---------------------------------------------------------------------------
class TypeRef:
    def __init__(self, width: int, is_float: bool, is_ptr: bool, signed: Optional[bool] = None):
        self.width = width
        self.is_float = is_float
        self.is_ptr = is_ptr
        self.signed = signed
    @staticmethod
    def unknown() -> 'TypeRef':
        return TypeRef(0, False, False, None)
    @staticmethod
    def from_string(s: Optional[str]) -> 'TypeRef':
        if not s or s == 'unknown':
            return TypeRef.unknown()
        if s == 'ptr' or s.startswith('i8*'):
            return TypeRef(64, False, True, None)
        if s == 'void':
            return TypeRef(0, False, False, None)
        if s in ('f32', 'float'):
            return TypeRef(32, True, False, None)
        if s in ('f64', 'double'):
            return TypeRef(64, True, False, None)

        signed = None
        base = s
        if s.endswith('_signed'):
            signed = True
            base = s[:-7]
        elif s.endswith('_unsigned'):
            signed = False
            base = s[:-9]

        if base.startswith('i') and base[1:].isdigit():
            return TypeRef(int(base[1:]), False, False, signed)
        return TypeRef.unknown()
    def to_string(self) -> str:
        if self.is_ptr:
            return 'ptr'
        if self.is_float:
            return 'float' if self.width == 32 else 'double'

        suffix = ''
        if self.signed is True:
            suffix = '_signed'
        elif self.signed is False:
            suffix = '_unsigned'

        return f"i{self.width}{suffix}" if self.width > 0 else 'unknown'
    def is_unknown(self) -> bool:
        return self.width == 0 and not self.is_ptr and not self.is_float and self.signed is None
def meet(t1: Optional[TypeRef], t2: Optional[TypeRef]) -> TypeRef:
    """Lattice LUB (widen on conflict). Retains ptr on i64 conflict per spec."""
    if t1 is None or t1.is_unknown():
        return t2 or TypeRef.unknown()
    if t2 is None or t2.is_unknown():
        return t1

    if t1.is_ptr and t2.is_ptr:
        return TypeRef(64, False, True, None)
    if t1.is_ptr != t2.is_ptr:
        return TypeRef(64, False, True, None) # conservative ptr semantics

    if t1.is_float and t2.is_float:
        return t1 if t1.width == t2.width else TypeRef.unknown()
    if t1.is_float != t2.is_float:
        return TypeRef.unknown()

    new_width = max(t1.width or 64, t2.width or 64)
    new_signed = t1.signed if t1.signed == t2.signed else None
    return TypeRef(new_width, False, False, new_signed)
# ---------------------------------------------------------------------------
# ABI / External helpers
# ---------------------------------------------------------------------------
INT_ARGS = ['RDI', 'RSI', 'RDX', 'RCX', 'R8', 'R9']
XMM_ARGS = [f'XMM{i}' for i in range(8)]
def get_canonical_reg(reg: str) -> Optional[str]:
    if not reg:
        return None
    r = reg.upper()
    map_rules = {
        'RAX': 'RAX', 'EAX': 'RAX', 'AX': 'RAX', 'AL': 'RAX',
        'RBX': 'RBX', 'EBX': 'RBX', 'BX': 'RBX', 'BL': 'RBX',
        'RCX': 'RCX', 'ECX': 'RCX', 'CX': 'RCX', 'CL': 'RCX',
        'RDX': 'RDX', 'EDX': 'RDX', 'DX': 'RDX', 'DL': 'RDX',
        'RSI': 'RSI', 'ESI': 'RSI', 'SI': 'RSI', 'SIL': 'RSI',
        'RDI': 'RDI', 'EDI': 'RDI', 'DI': 'RDI', 'DIL': 'RDI',
        'R8': 'R8', 'R8D': 'R8', 'R8W': 'R8', 'R8B': 'R8',
        'R9': 'R9', 'R9D': 'R9', 'R9W': 'R9', 'R9B': 'R9',
        'RSP': 'RSP', 'RBP': 'RBP'
    }
    return map_rules.get(r) or (r if r.startswith('XMM') else None)
class ExternalABIDB:
    KNOWN = {
        'printf': {'ret': TypeRef(32, False, False, True), 'args': [TypeRef(0, False, True, None)]},
        'fprintf': {'ret': TypeRef(32, False, False, True), 'args': [TypeRef(0, False, True, None), TypeRef(0, False, True, None)]},
        'scanf': {'ret': TypeRef(32, False, False, True), 'args': [TypeRef(0, False, True, None)]},
        'puts': {'ret': TypeRef(32, False, False, True), 'args': [TypeRef(0, False, True, None)]},
        'malloc': {'ret': TypeRef(0, False, True, None), 'args': [TypeRef(64, False, False, True)]},
        'free': {'ret': TypeRef(0, False, False, None), 'args': [TypeRef(0, False, True, None)]},
        'exit': {'ret': None, 'args': [TypeRef(32, False, False, True)], 'noreturn': True},
        'abort': {'ret': None, 'args': [], 'noreturn': True},
        'sqrt': {'ret': TypeRef(64, True, False, None), 'args': [TypeRef(64, True, False, None)]},
    }
    @staticmethod
    def get(name: str) -> Optional[Dict]:
        return ExternalABIDB.KNOWN.get(name)
class CallSite:
    def __init__(self, caller: Function, instr: Instruction, target_name: str, is_external: bool):
        self.caller = caller
        self.instr = instr
        self.target_name = target_name
        self.is_external = is_external
class InterproceduralAnalyzer:
    def __init__(self, program: Program):
        self.program = program
        self.functions: Dict[str, Function] = {}
        self.call_graph: List[CallSite] = []
        self.reverse_call_graph: Dict[str, List[CallSite]] = defaultdict(list)
        self.func_constraints: Dict[str, Dict] = defaultdict(
            lambda: {'ret': TypeRef.unknown(), 'args': [TypeRef.unknown() for _ in range(6)]}
        )
        self._collect_functions()
    def _collect_functions(self):
        for sec in self.program.sections:
            for child in sec.children:
                if isinstance(child, Function):
                    self.functions[child.id] = child
                    # Step 7 ABI seeds
                    arg_constraints = [TypeRef.unknown() for _ in range(6)]
                    for arg in child.arguments:
                        if arg.kind == 'register':
                            try:
                                idx = INT_ARGS.index(arg.location.upper())
                                t = TypeRef.from_string(arg.inferred_type)
                                if idx < len(arg_constraints):
                                    arg_constraints[idx] = t
                            except ValueError:
                                pass
                    self.func_constraints[child.id]['args'] = arg_constraints
                    # Respect explicit 'void' from Step 7 ABI recovery
                    # (critical for bare-RET ELF ctors/dtors and other void funcs).
                    # Prevents default-to-i64 heuristic when there is zero evidence
                    # of a return value (no RAX definition, no internal callers).
                    # Guarantees preservation of standard-enforced signatures.
                    if child.return_type == 'void':
                        self.func_constraints[child.id]['ret'] = TypeRef(0, False, False, None)
                    else:
                        rt = TypeRef.from_string(child.return_type)
                        self.func_constraints[child.id]['ret'] = rt if not rt.is_unknown() else TypeRef(64, False, False, None)
    def build_call_graph(self):
        for func in self.functions.values():
            for bb in func.basic_blocks:
                for instr in bb.instructions:
                    if instr.opcode.upper() == 'CALL' and instr.operands:
                        target = instr.operands[0]
                        name = target.name or (target.symbol_ref.name if getattr(target, 'symbol_ref', None) else None)
                        if name:
                            is_ext = name not in {f.entry_label for f in self.functions.values()}
                            cs = CallSite(func, instr, name, is_ext)
                            self.call_graph.append(cs)
                            self.reverse_call_graph[name].append(cs)
    def _find_type_passed_at_call(self, cs: CallSite, arg_idx: int) -> TypeRef:
        if arg_idx < len(INT_ARGS):
            target_reg = INT_ARGS[arg_idx]
        elif arg_idx < len(XMM_ARGS):
            target_reg = XMM_ARGS[arg_idx]
        else:
            return TypeRef.unknown()
        bb = cs.instr.parent
        if not bb:
            return TypeRef.unknown()

        try:
            call_idx = bb.instructions.index(cs.instr)
        except ValueError:
            return TypeRef.unknown()
        for i in range(call_idx - 1, -1, -1):
            prev = bb.instructions[i]
            if prev.operands and prev.operands[0].register:
                if get_canonical_reg(prev.operands[0].register) == target_reg:
                    if prev.op_refinement:
                        t = TypeRef.from_string(prev.op_refinement)
                        return t if not (target_reg.startswith('XMM') and not t.is_float) else TypeRef(64, True, False, None)

        return TypeRef.unknown()
    def _find_return_usage_at_call(self, cs: CallSite) -> TypeRef:
        bb = cs.instr.parent
        if not bb:
            return TypeRef.unknown()

        try:
            call_idx = bb.instructions.index(cs.instr)
        except ValueError:
            return TypeRef.unknown()
        for i in range(call_idx + 1, len(bb.instructions)):
            nxt = bb.instructions[i]
            for op in nxt.operands:
                if op.register:
                    canon = get_canonical_reg(op.register)
                    if canon == 'RAX' or op.register.upper().startswith('XMM0'):
                        if nxt.op_refinement:
                            t = TypeRef.from_string(nxt.op_refinement)
                            return t if not op.register.upper().startswith('XMM') else (t if t.is_float else TypeRef(64, True, False, None))

            if nxt.opcode.upper() in ('RET', 'JMP', 'CALL'):
                break

        return TypeRef.unknown()
    def _prune_noreturn_fallthrough(self):
        """Prune spurious fall-through after calls to (newly) noreturn functions."""
        noreturn_targets = {name for name, info in ExternalABIDB.KNOWN.items() if info.get('noreturn')}

        for f in self.functions.values():
            if (f.noreturn_kind == 'noreturn' or
                (f.lifted_signature and 'noreturn' in f.lifted_signature.attributes)):
                if f.entry_label:
                    noreturn_targets.add(f.entry_label)
        for cs in self.call_graph:
            if cs.target_name in noreturn_targets:
                bb = cs.instr.parent
                if bb and bb.successors:
                    old = list(bb.successors)
                    bb.successors.clear()
                    for succ in old:
                        if bb in getattr(succ, 'predecessors', []):
                            succ.predecessors.remove(bb)
    def run(self):
        self.build_call_graph()
        worklist = set(self.functions.keys())
        iteration = 0

        while worklist and iteration < 30: # safe bound
            iteration += 1
            next_worklist = set()

            for f_id in list(worklist):
                func = self.functions[f_id]
                changed = False
                callers = self.reverse_call_graph.get(func.entry_label, []) or self.reverse_call_graph.get(f_id, [])
                ext_sig = ExternalABIDB.get(func.entry_label)
                # Arg constraints (caller → callee)
                new_args = list(self.func_constraints[f_id]['args'])
                for cs in callers:
                    for i in range(6):
                        passed = self._find_type_passed_at_call(cs, i)
                        new_args[i] = meet(new_args[i], passed)

                if ext_sig and 'args' in ext_sig:
                    for i, at in enumerate(ext_sig['args']):
                        if i < len(new_args):
                            new_args[i] = meet(new_args[i], at)

                if new_args != self.func_constraints[f_id]['args']:
                    self.func_constraints[f_id]['args'] = new_args
                    changed = True
                # Return constraints (callee → caller + external)
                new_ret = self.func_constraints[f_id]['ret']
                for cs in callers:
                    usage = self._find_return_usage_at_call(cs)
                    new_ret = meet(new_ret, usage)

                if ext_sig:
                    if ext_sig.get('noreturn'):
                        new_ret = TypeRef.unknown()
                    elif ext_sig.get('ret'):
                        new_ret = meet(new_ret, ext_sig['ret'])

                new_ret = meet(new_ret, TypeRef.from_string(func.return_type))

                if new_ret != self.func_constraints[f_id]['ret']:
                    self.func_constraints[f_id]['ret'] = new_ret
                    changed = True
                if changed:
                    next_worklist.add(f_id)
                    for cs in callers:
                        next_worklist.add(cs.caller.id)
                    # Re-queue callees for arg flow
                    for cs_out in self.call_graph:
                        if cs_out.caller.id == f_id and cs_out.target_name in self.functions:
                            next_worklist.add(cs_out.target_name)

            worklist = next_worklist
        self._finalize_signatures()
        self._propagate_refinements()
        self._prune_noreturn_fallthrough()
    def _refine_external_abi_signature(self, func: Function, refined_ret: str) -> str:
        """Merge interprocedural results into Step 7 external sig while preserving structure."""
        existing = func.external_abi_signature
        if not existing or '(' not in existing:
            return f"{refined_ret} ()"
        _, rest = existing.split(' (', 1)
        return f"{refined_ret} ({rest}"
    def _propagate_refinements(self):
        """Minimal practical propagation to AST (args, return_type, op_refinement seeds)."""
        for f_id, cons in self.func_constraints.items():
            func = self.functions.get(f_id)
            if not func:
                continue
            for i, at in enumerate(cons['args']):
                if i < len(func.arguments) and not at.is_unknown():
                    func.arguments[i].inferred_type = at.to_string()
            rt = cons['ret']
            if not rt.is_unknown() and func.return_type != 'void':
                func.return_type = rt.to_string()
    def _finalize_signatures(self):
        """Finalize dual signatures.
        - lifted_signature: internal LLVM view (void + noreturn attribute when applicable).
        - external_abi_signature: boundary functions only – preserves the exact C ABI contract
          expected by the runtime/linker (return type is NEVER forced to void by noreturn).
        """
        for f_id, func in self.functions.items():
            cons = self.func_constraints[f_id]
            ext_sig = ExternalABIDB.get(func.entry_label)
            is_noreturn = (func.noreturn_kind == 'noreturn' or
                           (ext_sig and ext_sig.get('noreturn')))
            ret_t = cons['ret']
            # Base return type from interprocedural refinement + original Step-7 ABI seed
            base_ret = (ret_t.to_string() if not ret_t.is_unknown()
                        else (func.return_type or 'i64'))
            if base_ret.endswith(('_signed', '_unsigned')):
                base_ret = base_ret.rsplit('_', 1)[0]
            if base_ret == 'ptr':
                base_ret = 'i8*'
            # Lifted (internal) signature – noreturn becomes void + attribute
            lifted_ret_type_str = 'void' if is_noreturn else base_ret
            attributes = ['noreturn'] if is_noreturn else []
            func.lifted_signature = LiftedFunctionSignature(
                return_type=lifted_ret_type_str, attributes=attributes)
            # External ABI signature (boundary only)
            # MUST keep the original ABI return type (e.g. i32 for main)
            if func.is_boundary:
                if func.external_abi_signature and '(' in func.external_abi_signature:
                    # Step-7 ABI recovery is authoritative for the external contract
                    abi_ret_type_str = func.external_abi_signature.split(' (', 1)[0]
                else:
                    abi_ret_type_str = base_ret
                func.external_abi_signature = self._refine_external_abi_signature(
                    func, abi_ret_type_str)
def main():
    args = sys.argv[1:]

    # Handle optional --print flag
    print_proto = False
    if "--print" in args:
        print_proto = True
        args.remove("--print")

    try:
        raw = sys.stdin.read()
        if not raw.strip():
            sys.stderr.write("No input on stdin.\n")
            sys.exit(2)
        obj = json.loads(raw)
    except Exception as e:
        sys.stderr.write(f"JSON error: {e}\n")
        sys.exit(3)

    try:
        # 1. Deserialize via AST nodes
        program = legacy_program_dict_to_ast(obj, include_enhancements=True)

        # 2. Perform step 9 refinements directly on the AST
        analyzer = InterproceduralAnalyzer(program)
        analyzer.run()

        # 3. Serialize output directly to Protobuf
        proto_msg = ast_to_proto(program, obj)

        # Dump human readable format to standard out if requested
        if print_proto:
            print(MessageToJson(proto_msg, indent=2))

        # Determine output location (file or stdout)
        out_path = args[0] if len(args) > 0 else None
        if out_path:
            with open(out_path, "wb") as f:
                f.write(proto_msg.SerializeToString(deterministic=True))
            sys.stderr.write(f"✅ Serialized {len(proto_msg.SerializeToString()):,} bytes → {out_path}\n")
        else:
            # Output directly to stdout buffer to continue the pipeline cleanly (skipping text serialization)
            if not print_proto:
                sys.stdout.buffer.write(proto_msg.SerializeToString(deterministic=True))

    except Exception as e:
        sys.stderr.write(f"Step 9 error: {repr(e)}\n")
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
if __name__ == "__main__":
    main()
