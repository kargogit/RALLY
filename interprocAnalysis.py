"""
interprocAnal.py
Step 9: Interprocedural Propagation and Signature Finalization.
Performs interprocedural analysis to refine operation widths, signedness,
and pointer distinctions across function boundaries. It finalizes LLVM
function signatures (ret_type (%State*)) and attributes like 'noreturn'.
Inputs: Enriched AST from Step 8 (via stdin JSON).
Outputs: Enriched AST with finalized 'lifted_signature' on each Function
 and propagated refinements (via stdout JSON).
"""
import sys
import json
import copy
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import defaultdict
# Assuming astNodes.py is available in the environment or path
from astNodes import (
    legacy_program_dict_to_ast,
    ast_to_legacy_program_dict,
    Program,
    Function,
    BasicBlock,
    Instruction,
    Operand,
    LiftedFunctionSignature
)
# ---------------------------------------------------------------------------
# Type System (Reused/Adapted from Step 8)
# ---------------------------------------------------------------------------
class TypeRef:
    """Represents a type constraint (width, signedness, float, ptr)."""
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
        if not s:
            return TypeRef.unknown()
        if s == 'ptr':
            return TypeRef(0, False, True, None)
        if s == 'void':
            return TypeRef(0, False, False, None)  # Special case
        if s == 'f32':
            return TypeRef(32, True, False, None)
        if s == 'f64':
            return TypeRef(64, True, False, None)
        base = s
        signed = None
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
        if self.is_ptr: return 'ptr'
        if self.is_float:
            return 'float' if self.width == 32 else 'double'
        suffix = ''
        if self.signed is True: suffix = '_signed'
        elif self.signed is False: suffix = '_unsigned'
        return f"i{self.width}{suffix}"

    def __repr__(self):
        return self.to_string()

    def is_unknown(self):
        return not self.is_ptr and not self.is_float and self.width == 0 and self.signed is None

def meet(t1: Optional[TypeRef], t2: Optional[TypeRef]) -> TypeRef:
    """Lattice meet: returns the least upper bound (widest safe type)."""
    if t1 is None: return t2 if t2 else TypeRef.unknown()
    if t2 is None: return t1
    # Handle void/unknown as top
    if t1.is_unknown(): return t2
    if t2.is_unknown(): return t1
    # If one is ptr and other is not, we can't safely merge (unless we treat ptr as i64, but here we distinguish)
    if t1.is_ptr != t2.is_ptr:
        return TypeRef.unknown()
    if t1.is_ptr and t2.is_ptr:
        return t1  # pointers are consistent
    if t1.is_float and t2.is_float:
        return t1 if t1.width == t2.width else TypeRef.unknown()
    if t1.is_float != t2.is_float:
        return TypeRef.unknown()
    # Integer merge
    # Widen width
    new_width = max(t1.width, t2.width)
    # Merge signedness: if conflict, drop signedness (unknown signedness)
    new_signed = None
    if t1.signed == t2.signed:
        new_signed = t1.signed
    return TypeRef(new_width, False, False, new_signed)

# ---------------------------------------------------------------------------
# System V ABI / External Function Helpers
# ---------------------------------------------------------------------------
# Map of canonical register names to argument indices (System V AMD64)
INT_ARGS = ['RDI', 'RSI', 'RDX', 'RCX', 'R8', 'R9']
XMM_ARGS = ['XMM0', 'XMM1', 'XMM2', 'XMM3', 'XMM4', 'XMM5', 'XMM6', 'XMM7']

def get_canonical_reg(reg: str) -> Optional[str]:
    if not reg: return None
    r = reg.upper()
    # Simple mapping for standard regs
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
    if r in map_rules: return map_rules[r]
    if r.startswith('XMM'): return r  # XMM0-15
    return None

class ExternalABIDB:
    """Knowledge base for known external functions."""
    # Known signatures: (ret_type, [arg_types...])
    # None for ret means void/noreturn usually, specific check required
    KNOWN = {
        'printf': {'ret': TypeRef(32, False, False, True), 'args': [TypeRef(0, False, True, None)]},  # i32 (i8*, ...) - variadic
        'fprintf': {'ret': TypeRef(32, False, False, True), 'args': [TypeRef(0, False, True, None), TypeRef(0, False, True, None)]},
        'scanf': {'ret': TypeRef(32, False, False, True), 'args': [TypeRef(0, False, True, None)]},
        'puts': {'ret': TypeRef(32, False, False, True), 'args': [TypeRef(0, False, True, None)]},
        'malloc': {'ret': TypeRef(0, False, True, None), 'args': [TypeRef(64, False, False, True)]},  # void* (size_t)
        'free': {'ret': TypeRef(0, False, False, None), 'args': [TypeRef(0, False, True, None)]},  # void (void*)
        'exit': {'ret': None, 'args': [TypeRef(32, False, False, True)], 'noreturn': True},  # void (i32)
        'abort': {'ret': None, 'args': [], 'noreturn': True},
        'sqrt': {'ret': TypeRef(64, True, False, None), 'args': [TypeRef(64, True, False, None)]},  # double (double)
    }

    @staticmethod
    def get(name: str) -> Optional[Dict]:
        return ExternalABIDB.KNOWN.get(name)

# ---------------------------------------------------------------------------
# Analysis Logic
# ---------------------------------------------------------------------------
class CallSite:
    def __init__(self, caller: Function, instr: Instruction, target_name: str, is_external: bool):
        self.caller = caller
        self.instr = instr
        self.target_name = target_name  # Name of function (string)
        self.is_external = is_external

class InterproceduralAnalyzer:
    def __init__(self, program: Program):
        self.program = program
        self.functions: Dict[str, Function] = {}  # id -> Function
        self.call_graph: List[CallSite] = []  # List of all calls
        self.reverse_call_graph: Dict[str, List[CallSite]] = defaultdict(list)  # target_name -> CallSites
        # Constraints per function (accumulated during fixed point)
        self.func_constraints: Dict[str, Dict] = defaultdict(lambda: {
            'ret': TypeRef.unknown(),
            'args': []  # List of TypeRef
        })
        self._collect_functions()

    def _collect_functions(self):
        for sec in self.program.sections:
            for child in sec.children:
                if isinstance(child, Function):
                    self.functions[child.id] = child
                    # Initialize constraints from Step 7 ABI info if present
                    arg_constraints = []
                    for i in range(6):  # Cap at 6 standard regs for initialization
                        arg_constraints.append(TypeRef.unknown())
                    # Override with Step 7 specific types if available
                    if child.arguments:
                        for arg in child.arguments:
                            if arg.kind == 'register':
                                # Map to index if possible
                                try:
                                    idx = INT_ARGS.index(arg.location.upper())
                                    t = TypeRef.from_string(arg.inferred_type)
                                    if t.is_unknown(): t = TypeRef(64, False, False, None)
                                    if idx < len(arg_constraints):
                                        arg_constraints[idx] = t
                                except ValueError:
                                    pass
                    self.func_constraints[child.id]['args'] = arg_constraints
                    # Init ret type
                    rt = TypeRef.from_string(child.return_type)
                    if rt.is_unknown() and not child.noreturn_kind and child.return_type != 'void':
                        rt = TypeRef(64, False, False, None)  # Default i64
                    self.func_constraints[child.id]['ret'] = rt

    def build_call_graph(self):
        for func in self.functions.values():
            for bb in func.basic_blocks:
                for instr in bb.instructions:
                    if instr.opcode.upper() == 'CALL':
                        if not instr.operands:
                            continue
                        # Target is typically op 0
                        target = instr.operands[0]
                        name = None
                        is_ext = False
                        if target.name:
                            name = target.name
                            # Check if internal
                            found = False
                            for f_id, f_obj in self.functions.items():
                                if f_obj.entry_label == name:
                                    name = f_obj.entry_label  # Normalize to entry label
                                    found = True
                                    break
                            if not found:
                                is_ext = True
                        elif target.symbol_ref:
                            name = target.symbol_ref.name
                            # Check if internal label
                            if any(f.entry_label == name for f in self.functions.values()):
                                is_ext = False
                            else:
                                is_ext = True
                        if name:
                            cs = CallSite(func, instr, name, is_ext)
                            self.call_graph.append(cs)
                            self.reverse_call_graph[name].append(cs)

    def _find_type_passed_at_call(self, cs: CallSite, arg_idx: int) -> Optional[TypeRef]:
        """Heuristic: Scan backwards in the BasicBlock to find the instruction that sets the register corresponding to arg_idx."""
        target_reg = None
        if arg_idx < len(INT_ARGS):
            target_reg = INT_ARGS[arg_idx]
        else:
            return TypeRef.unknown()
        actual_bb = cs.instr.parent
        if not actual_bb: return TypeRef.unknown()
        instr_list = actual_bb.instructions
        try:
            call_idx = instr_list.index(cs.instr)
        except ValueError:
            return TypeRef.unknown()
        for i in range(call_idx - 1, -1, -1):
            prev_instr = instr_list[i]
            if prev_instr.operands and len(prev_instr.operands) > 0:
                dst = prev_instr.operands[0]
                if dst.register and get_canonical_reg(dst.register) == target_reg:
                    if prev_instr.op_refinement:
                        return TypeRef.from_string(prev_instr.op_refinement)
                    w = 64
                    r = dst.register.upper()
                    if 'D' in r and r not in ['RDI', 'RDX']: w = 32
                    elif 'W' in r: w = 16
                    elif 'B' in r or 'L' in r: w = 8
                    return TypeRef(w, False, False, None)
                if target_reg.startswith('XMM') and dst.register:
                    if dst.register.upper() == target_reg:
                        if prev_instr.op_refinement:
                            return TypeRef.from_string(prev_instr.op_refinement)
                        return TypeRef(128, True, False, None)
        # Not found in immediate block. Check caller's args
        if arg_idx < len(cs.caller.arguments):
            arg_desc = cs.caller.arguments[arg_idx]
            return TypeRef.from_string(arg_desc.inferred_type)
        return TypeRef.unknown()

    def _find_return_usage_at_call(self, cs: CallSite) -> Optional[TypeRef]:
        """Scan forwards from the call to see how RAX (or XMM0) is used."""
        actual_bb = cs.instr.parent
        if not actual_bb: return TypeRef.unknown()
        instr_list = actual_bb.instructions
        try:
            call_idx = instr_list.index(cs.instr)
        except ValueError:
            return TypeRef.unknown()
        for i in range(call_idx + 1, len(instr_list)):
            next_instr = instr_list[i]
            uses_ret = False
            for op in next_instr.operands:
                if op.register:
                    canon = get_canonical_reg(op.register)
                    if canon == 'RAX' or op.register.upper() == 'XMM0':
                        uses_ret = True
                        if op == next_instr.operands[0]:  # Is def
                            if next_instr.op_refinement:
                                return TypeRef.from_string(next_instr.op_refinement)
                            w = 64
                            if 'D' in op.register.upper(): w = 32
                            elif 'W' in op.register.upper(): w = 16
                            elif 'B' in op.register.upper(): w = 8
                            return TypeRef(w, False, False, None)
            if uses_ret:
                return TypeRef(64, False, False, None)
            if next_instr.opcode.upper() in ['RET', 'JMP', 'CALL']:
                break
        return TypeRef.unknown()

    def run(self):
        self.build_call_graph()
        # Fixed point iteration
        worklist = set(self.functions.keys())
        iteration = 0
        while worklist and iteration < 20:  # Limit iterations just in case
            iteration += 1
            next_worklist = set()
            for f_id, func in self.functions.items():
                changed = False
                callers = self.reverse_call_graph.get(func.entry_label, [])
                if not callers and func.entry_label not in self.reverse_call_graph:
                    callers = self.reverse_call_graph.get(f_id, [])
                ext_sig = ExternalABIDB.get(func.entry_label)
                # --- Merge Arguments Constraints ---
                current_args = self.func_constraints[f_id]['args']
                new_args = list(current_args)
                for cs in callers:
                    for i in range(6):
                        passed_t = self._find_type_passed_at_call(cs, i)
                        new_args[i] = meet(new_args[i], passed_t)
                if ext_sig:
                    for i, arg_t in enumerate(ext_sig['args']):
                        if i < len(new_args):
                            new_args[i] = meet(new_args[i], arg_t)
                if new_args != current_args:
                    self.func_constraints[f_id]['args'] = new_args
                    changed = True
                # --- Merge Return Constraints ---
                current_ret = self.func_constraints[f_id]['ret']
                new_ret = current_ret
                for cs in callers:
                    usage_t = self._find_return_usage_at_call(cs)
                    new_ret = meet(new_ret, usage_t)
                if ext_sig:
                    if ext_sig.get('noreturn'):
                        new_ret = TypeRef.unknown()
                    elif ext_sig['ret']:
                        new_ret = meet(new_ret, ext_sig['ret'])
                step8_ret = TypeRef.from_string(func.return_type)
                new_ret = meet(new_ret, step8_ret)
                if new_ret != current_ret:
                    self.func_constraints[f_id]['ret'] = new_ret
                    changed = True
                if changed:
                    for cs in callers:
                        next_worklist.add(cs.caller.id)
            worklist = next_worklist
        self._finalize_signatures()

    def _finalize_signatures(self):
        for f_id, func in self.functions.items():
            constraints = self.func_constraints[f_id]
            # Determine Return Type String
            ret_type_str = 'void'
            is_noreturn = func.noreturn_kind == 'noreturn'
            ext_sig = ExternalABIDB.get(func.entry_label)
            if ext_sig and ext_sig.get('noreturn'):
                is_noreturn = True
            ret_t = constraints['ret']
            if is_noreturn:
                ret_type_str = 'void'
            elif not ret_t.is_unknown():
                ret_type_str = ret_t.to_string()
            else:
                # Fallback to Step 7 or i64
                if func.return_type:
                    ret_type_str = func.return_type
                else:
                    ret_type_str = 'i64'
            # Adjust naming for LLVM
            if ret_type_str.endswith('_signed') or ret_type_str.endswith('_unsigned'):
                ret_type_str = ret_type_str.rsplit('_', 1)[0]
            if ret_type_str == 'ptr':
                ret_type_str = 'i8*'
            if ret_type_str == 'f32': ret_type_str = 'float'
            if ret_type_str == 'f64': ret_type_str = 'double'
            attributes = []
            if is_noreturn:
                attributes.append('noreturn')
            func.lifted_signature = LiftedFunctionSignature(
                return_type=ret_type_str,
                attributes=attributes
            )

def main():
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            sys.stderr.write("No input on stdin.\n")
            sys.exit(2)
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"JSON Parse Error: {e}\n")
        sys.exit(3)
    try:
        program = legacy_program_dict_to_ast(obj, include_enhancements=True)
        analyzer = InterproceduralAnalyzer(program)
        analyzer.run()
        legacy_out = ast_to_legacy_program_dict(program, include_instr_locations=False, include_enhancements=True)
        json.dump(legacy_out, sys.stdout, indent=2)
        sys.stdout.write("\n")
    except Exception as e:
        sys.stderr.write(f"Error in Step 9: {repr(e)}\n")
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
