"""
typeInference.py
Step 8: Intraprocedural Data-Flow Analysis for Type Refinement.
Performs lightweight, fixed-point iteration over the CFG to infer and annotate
scalar operation widths, signedness, pointer usages, and basic scalar
floating-point types for operands, instructions, and stack slots.
Inputs: Enriched AST from Step 7 (via stdin JSON).
Outputs: Enriched AST with refinement annotations (via stdout JSON).
"""
import sys
import json
import copy
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict
from astNodes import (
    legacy_program_dict_to_ast,
    ast_to_legacy_program_dict,
    Program,
    Function,
    BasicBlock,
    Instruction,
    Operand,
    StackSlot,
    ArgumentDescriptor
)

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
        if not s:
            return TypeRef.unknown()
        if '*' in s:
            return TypeRef(0, False, True, None)
        if s == 'ptr':
            return TypeRef(0, False, True, None)
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
            return 'f32' if self.width == 32 else 'f64'
        suffix = ''
        if self.signed is True: suffix = '_signed'
        elif self.signed is False: suffix = '_unsigned'
        return f"i{self.width}{suffix}"

    def __repr__(self):
        return self.to_string()

    def __eq__(self, other):
        return (self.width == other.width and
                self.is_float == other.is_float and
                self.is_ptr == other.is_ptr and
                self.signed == other.signed)

    def is_unknown(self):
        return not self.is_ptr and not self.is_float and self.width == 0 and self.signed is None

def meet(t1: Optional[TypeRef], t2: Optional[TypeRef]) -> TypeRef:
    if t1 is None: return t2 if t2 else TypeRef.unknown()
    if t2 is None: return t1
    # NEW: Treat unknown as top – any conflict or unknown in predecessors → unknown
    if t1.is_unknown() or t2.is_unknown():
        return TypeRef.unknown()
    if t1.is_ptr and t2.is_ptr: return t1
    if t1.is_ptr != t2.is_ptr: return TypeRef.unknown()
    if t1.is_float and t2.is_float:
        return t1 if t1.width == t2.width else TypeRef.unknown()
    if t1.is_float != t2.is_float: return TypeRef.unknown()
    if t1.width == t2.width:
        if t1.signed == t2.signed: return t1
        return TypeRef(t1.width, False, False, None)
    else:
        return TypeRef.unknown()

def get_canonical_reg(reg: str) -> Optional[str]:
    r = reg.upper()
    if r in ['RAX', 'EAX', 'AX', 'AH', 'AL']: return 'RAX'
    if r in ['RBX', 'EBX', 'BX', 'BH', 'BL']: return 'RBX'
    if r in ['RCX', 'ECX', 'CX', 'CH', 'CL']: return 'RCX'
    if r in ['RDX', 'EDX', 'DX', 'DH', 'DL']: return 'RDX'
    if r in ['RSI', 'ESI', 'SI', 'SIL']: return 'RSI'
    if r in ['RDI', 'EDI', 'DI', 'DIL']: return 'RDI'
    if r in ['RBP', 'EBP', 'BP', 'BPL']: return 'RBP'
    if r in ['RSP', 'ESP', 'SP', 'SPL']: return 'RSP'
    if r in ['R8', 'R8D', 'R8W', 'R8B']: return 'R8'
    if r in ['R9', 'R9D', 'R9W', 'R9B']: return 'R9'
    if r in ['R10', 'R10D', 'R10W', 'R10B']: return 'R10'
    if r in ['R11', 'R11D', 'R11W', 'R11B']: return 'R11'
    if r in ['R12', 'R12D', 'R12W', 'R12B']: return 'R12'
    if r in ['R13', 'R13D', 'R13W', 'R13B']: return 'R13'
    if r in ['R14', 'R14D', 'R14W', 'R14B']: return 'R14'
    if r in ['R15', 'R15D', 'R15W', 'R15B']: return 'R15'
    if r.startswith('XMM'): return r
    return None

def get_reg_width(reg: str) -> Optional[int]:
    r = reg.upper()
    if r in ['AL', 'BL', 'CL', 'DL', 'SIL', 'DIL', 'BPL', 'SPL', 'R8B', 'R9B', 'R10B', 'R11B', 'R12B', 'R13B', 'R14B', 'R15B']: return 8
    if r in ['AH', 'BH', 'CH', 'DH']: return 8
    if r in ['AX', 'BX', 'CX', 'DX', 'SI', 'DI', 'BP', 'SP', 'R8W', 'R9W', 'R10W', 'R11W', 'R12W', 'R13W', 'R14W', 'R15W']: return 16
    if r in ['EAX', 'EBX', 'ECX', 'EDX', 'ESI', 'EDI', 'EBP', 'ESP', 'R8D', 'R9D', 'R10D', 'R11D', 'R12D', 'R13D', 'R14D', 'R15D']: return 32
    if r in ['RAX', 'RBX', 'RCX', 'RDX', 'RSI', 'RDI', 'RBP', 'RSP', 'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15']: return 64
    if r.startswith('XMM'): return 128
    return None

class RefinementAnalyzer:
    def __init__(self, program: Program):
        self.program = program
        self.symbol_types = self._extract_symbol_types()

    def _extract_symbol_types(self) -> Dict[str, str]:
        types = {}
        for name, data in self.program.symbol_table.items():
            if 'llvm_type' in data:
                types[name] = data['llvm_type']
        return types

    def run(self):
        for section in self.program.sections:
            for child in section.children:
                if isinstance(child, Function):
                    self._analyze_function(child)

    def _analyze_function(self, func: Function):
        if not func.basic_blocks:
            return

        worklist: List[BasicBlock] = list(func.basic_blocks)

        arg_types = {}
        for arg in func.arguments:
            if arg.kind == 'register':
                t = TypeRef.from_string(arg.inferred_type)
                if t.is_unknown():
                    t = TypeRef(64, False, False, None)
                arg_types[arg.location] = t

        stack_slots_map = {ss.offset: TypeRef.from_string(ss.inferred_type) for ss in func.stack_slots}

        states: Dict[str, Dict[str, TypeRef]] = {}

        while worklist:
            bb = worklist.pop(0)
            bb_id = bb.id
            new_state: Dict[str, TypeRef] = {}

            if not bb.predecessors:
                new_state['RSP'] = TypeRef(0, False, True, None)
                new_state['RBP'] = TypeRef(0, False, True, None)
                for reg, t in arg_types.items():
                    new_state[reg] = t
            else:
                pred_states = [states.get(p.id, {}) for p in bb.predecessors]
                if not pred_states:
                    continue
                all_regs = set()
                for ps in pred_states:
                    all_regs.update(ps.keys())
                for reg in all_regs:
                    merged_type = None
                    for ps in pred_states:
                        merged_type = meet(merged_type, ps.get(reg, TypeRef.unknown()))
                    if merged_type:
                        new_state[reg] = merged_type

            if bb_id in states and states[bb_id] == new_state:
                continue

            current_state = copy.deepcopy(new_state)
            for instr in bb.instructions:
                self._process_instruction(instr, current_state, stack_slots_map, arg_types)

            states[bb_id] = current_state
            for succ in bb.successors:
                if succ not in worklist:
                    worklist.append(succ)

    def _process_instruction(self, instr: Instruction, state: Dict[str, TypeRef], stack_map: Dict[int, TypeRef], arg_types: Dict[str, TypeRef]):
        opcode = instr.opcode.upper()
        operands = instr.operands
        op_ref = TypeRef.unknown()

        def get_operand_type(op: Operand) -> Optional[TypeRef]:
            if op.register:
                return state.get(get_canonical_reg(op.register))
            if op.integer:
                val_str = str(op.integer.value)
                size_str = op.size
                width = 64
                if size_str == 'BYTE': width = 8
                elif size_str == 'WORD': width = 16
                elif size_str == 'DWORD': width = 32
                elif size_str == 'QWORD': width = 64
                else:
                    try:
                        v = int(val_str, 0)
                        if -128 <= v <= 127: width = 8
                        elif -32768 <= v <= 32767: width = 16
                        elif -2**31 <= v <= 2**31-1: width = 32
                    except: pass
                signed = None
                try:
                    v = int(val_str, 0)
                    if v < 0: signed = True
                except: pass
                return TypeRef(width, False, False, signed)
            if op.memory:
                width = 64
                size_str = op.size
                if size_str == 'BYTE': width = 8
                elif size_str == 'WORD': width = 16
                elif size_str == 'DWORD': width = 32
                elif size_str == 'QWORD': width = 64
                if op.symbol_ref:
                    sym_type = self.symbol_types.get(op.symbol_ref.name)
                    if sym_type:
                        if 'i8*' in sym_type: return TypeRef(0, False, True, None)
                        if 'i8' in sym_type and '*' not in sym_type: return TypeRef(8, False, False, None)
                        if 'i16' in sym_type and '*' not in sym_type: return TypeRef(16, False, False, None)
                        if 'i32' in sym_type and '*' not in sym_type: return TypeRef(32, False, False, None)
                        if 'i64' in sym_type and '*' not in sym_type: return TypeRef(64, False, False, None)
                        if 'float' in sym_type: return TypeRef(32, True, False, None)
                        if 'double' in sym_type: return TypeRef(64, True, False, None)
                if op.memory.base in ['RBP', 'RSP'] and isinstance(op.memory.displacement, int):
                    off = op.memory.displacement
                    if off in stack_map:
                        st = stack_map[off]
                        if not st.is_unknown():
                            return st
                if op.memory.base:
                    canon = get_canonical_reg(op.memory.base)
                    if canon and canon in state and state[canon].is_ptr:
                        return TypeRef(0, False, True, None)
                if op.memory.index:
                    canon = get_canonical_reg(op.memory.index)
                    if canon and canon in state and state[canon].is_ptr:
                        return TypeRef(0, False, True, None)
                return TypeRef(width, False, False, None)
            if op.expression:
                if isinstance(op.expression, str):
                    return TypeRef(0, False, True, None)
                elif isinstance(op.expression, dict):
                    sub_types = []
                    exp_key = next(iter(op.expression))
                    subs = op.expression[exp_key]
                    for sub_dict in subs:
                        sub_op = Operand()
                        if 'register' in sub_dict:
                            sub_op.register = sub_dict['register']
                        sub_t = get_operand_type(sub_op)
                        sub_types.append(sub_t)
                    merged = None
                    for st in sub_types:
                        merged = meet(merged, st)
                    return merged if merged else TypeRef.unknown()
            # NEW: Direct symbol reference (e.g., MOV reg, count where count is equ constant)
            if (op.name or (op.symbol_ref and not op.memory)):
                sym_name = op.name or (op.symbol_ref.name if op.symbol_ref else None)
                if sym_name:
                    sym_data = self.program.symbol_table.get(sym_name, {})
                    if sym_data.get('kind') == 'constant':
                        lt = sym_data.get('llvm_type')
                        if lt:
                            return TypeRef.from_string(lt)
                        return TypeRef(64, False, False, None)
                    # Data symbol used directly (rare, treat as address)
                    return TypeRef(0, False, True, None)
            return TypeRef.unknown()

        def set_reg_type(reg_name: str, t: TypeRef):
            canon = get_canonical_reg(reg_name)
            if canon:
                state[canon] = t
                w = get_reg_width(reg_name)
                if w == 32:
                    state[canon] = t
                elif w < 32:
                    state[canon] = TypeRef.unknown()

        src_types = [get_operand_type(op) for op in operands]
        is_sse = any(op.register and op.register.startswith('XMM') for op in operands if op.register)

        if is_sse:
            ss_type = None
            if 'SS' in opcode: ss_type = 32
            elif 'SD' in opcode: ss_type = 64
            if ss_type in [32, 64]:
                t = TypeRef(ss_type, True, False, None)
                for op in operands:
                    if op.register and op.register.startswith('XMM'):
                        state[op.register] = t
                        op.inferred_type = t.to_string()
                op_ref = t

        dst_type = TypeRef.unknown()
        dst_reg = None
        dst_width = 64

        if len(operands) > 0 and operands[0].register:
            dst_reg = operands[0].register
            dst_width = get_reg_width(dst_reg) or 64
            if opcode in ['MOV', 'MOVQ', 'MOVDQU']:
                if len(src_types) > 1 and src_types[1]:
                    dst_type = src_types[1]
                # If source unknown, fall back to plain integer of destination width
                if dst_type.is_unknown():
                    dst_type = TypeRef(dst_width, False, False, None)
                # Force operation width to destination register width
                if dst_type.is_ptr:
                    dst_type.width = 64 # pointers are 64-bit in this environment
                elif not dst_type.is_float:
                    dst_type.width = dst_width
            elif opcode.startswith('CMOV'):
                if len(src_types) > 1 and src_types[1]:
                    dst_type = src_types[1]
                if dst_type.is_unknown():
                    dst_type = TypeRef(dst_width, False, False, None)
                if dst_type.is_ptr:
                    dst_type.width = 64
                elif not dst_type.is_float:
                    dst_type.width = dst_width
            elif opcode in ['ADD', 'SUB', 'AND', 'OR', 'XOR', 'ADC', 'SBB']:
                t0 = get_operand_type(operands[0])
                t1 = src_types[1] if len(src_types) > 1 else None
                merged_t = meet(t0, t1)
                if merged_t.is_unknown():
                    merged_t = TypeRef(dst_width, False, False, None)
                else:
                    merged_t.width = dst_width
                dst_type = merged_t
                op_ref = dst_type
            elif opcode in ['SHL', 'SAL', 'SAR', 'SHR']:
                signed = get_operand_type(operands[0]).signed
                if opcode in ['SAR']: signed = True
                elif opcode in ['SHR']: signed = False
                dst_type = TypeRef(dst_width, False, False, signed)
                op_ref = dst_type
            elif opcode in ['INC', 'DEC', 'NEG', 'NOT']:
                t0 = get_operand_type(operands[0])
                is_ptr = t0.is_ptr if t0 else False
                signed = t0.signed if t0 else None
                dst_type = TypeRef(dst_width, False, is_ptr, signed)
                op_ref = dst_type
            elif opcode == 'IMUL':
                dst_type = TypeRef(dst_width, False, False, True)
                op_ref = dst_type
            elif opcode == 'MUL':
                dst_type = TypeRef(dst_width, False, False, False)
                op_ref = dst_type
            elif opcode in ['MOVSX', 'MOVSXD']:
                dst_type = TypeRef(dst_width, False, False, True)
                op_ref = dst_type
                src_op = operands[1] if len(operands) > 1 else None
                if src_op and (src_op.inferred_type is None or TypeRef.from_string(src_op.inferred_type).is_unknown()):
                    src_width = {'BYTE':8, 'WORD':16, 'DWORD':32}.get(src_op.size, 32)
                    src_op.inferred_type = TypeRef(src_width, False, False, True).to_string()
            elif opcode in ['MOVZX']:
                dst_type = TypeRef(dst_width, False, False, False)
                op_ref = dst_type
                src_op = operands[1] if len(operands) > 1 else None
                if src_op and (src_op.inferred_type is None or TypeRef.from_string(src_op.inferred_type).is_unknown()):
                    src_width = {'BYTE':8, 'WORD':16, 'DWORD':32}.get(src_op.size, 32)
                    src_op.inferred_type = TypeRef(src_width, False, False, False).to_string()
            elif opcode == 'LEA':
                dst_type = TypeRef(0, False, True, None)  # Seed as pointer
                if len(operands) > 1:
                    src_op = operands[1]
                    # Address computation cases → keep as ptr
                    if src_op.memory or src_op.symbol_ref or (src_op.expression and isinstance(src_op.expression, str)):
                        pass  # remain ptr
                    else:
                        # Potential pure arithmetic (register-only expression)
                        src_t = get_operand_type(src_op)
                        if src_t and not src_t.is_ptr and not src_t.is_unknown() and not src_t.is_float:
                            signed = src_t.signed
                            dst_type = TypeRef(dst_width, False, False, signed)
                op_ref = dst_type
            elif opcode in ['CVTTSS2SI', 'CVTTS2SI']:
                dst_type = TypeRef(dst_width, False, False, True)
                op_ref = dst_type
            elif opcode.startswith('SET'):
                dst_type = TypeRef(8, False, False, None)
                op_ref = dst_type
            elif opcode in ['IDIV', 'DIV']:
                signed = (opcode == 'IDIV')
                width = dst_width
                if operands:
                    op0 = operands[0]
                    if op0.size:
                        op_sz = op0.size
                        if op_sz == 'QWORD': width = 64
                        elif op_sz == 'DWORD': width = 32
                        elif op_sz == 'WORD': width = 16
                        elif op_sz == 'BYTE': width = 8
                    state['RAX'] = TypeRef(width, False, False, signed)
                    state['RDX'] = TypeRef(width, False, False, signed)
                op_ref = TypeRef(width, False, False, signed)

            if dst_type and not dst_type.is_unknown():
                set_reg_type(dst_reg, dst_type)
                if op_ref.is_unknown():
                    op_ref = dst_type

        # Handle instructions without explicit destination register (moved outside)
        if opcode in ['CDQ', 'CQO']:
            width = 32 if opcode == 'CDQ' else 64
            op_ref = TypeRef(width, False, False, True)
            state['RAX'] = TypeRef(width, False, False, True)
            state['RDX'] = TypeRef(width, False, False, True)
        
        elif opcode == 'XCHG':
            width = 64
            sz_map = {'BYTE':8, 'WORD':16, 'DWORD':32, 'QWORD':64}
            sz = None
            if operands:
                sz = operands[0].size or (operands[1].size if len(operands) > 1 else None)
                if sz:
                    width = sz_map.get(sz, 64)
                else:
                    for opi in operands:
                        if opi.register:
                            width = get_reg_width(opi.register) or 64
                            break
            t = TypeRef(width, False, False, None)
            op_ref = t
            for opi in operands:
                if opi.register:
                    set_reg_type(opi.register, t)
                if opi.memory:
                    opi.inferred_type = t.to_string()
        
        elif opcode == 'LOOP':
            # LOOP implicitly uses RCX as a counter.
            # In 64-bit mode, it decrements RCX. We should update RCX state.
            rcx_type = state.get('RCX')
            if rcx_type is None or rcx_type.is_unknown():
                # Default to 64-bit counter in 64-bit mode
                rcx_type = TypeRef(64, False, False, None)
            
            op_ref = rcx_type
            state['RCX'] = rcx_type

        # Special handling for LOCK-prefixed ops on memory
        if opcode in ['INC', 'DEC', 'NEG', 'NOT'] and instr.prefix and instr.prefix.upper() == 'LOCK':
            if operands and operands[0].memory:
                width = 64
                if operands[0].size:
                    width = {'BYTE':8, 'WORD':16, 'DWORD':32, 'QWORD':64}.get(operands[0].size, 64)
                t0 = get_operand_type(operands[0])
                is_ptr = t0.is_ptr if t0 else False
                signed = t0.signed if t0 else None
                op_ref = TypeRef(width, False, is_ptr, signed)
                operands[0].inferred_type = op_ref.to_string()

        if opcode in ['CMP', 'TEST']:
            t0 = src_types[0] if len(src_types) > 0 else None
            t1 = src_types[1] if len(src_types) > 1 else None
            merged_t = meet(t0, t1)
            width = None
            for opi in operands[:2]:
                if opi.register:
                    width = get_reg_width(opi.register)
                    break
                elif opi.memory and opi.size:
                    width = {'BYTE':8, 'WORD':16, 'DWORD':32, 'QWORD':64}.get(opi.size)
                    break
                elif opi.size:
                    width = {'BYTE':8, 'WORD':16, 'DWORD':32, 'QWORD':64}.get(opi.size)
                    break
            if width is None:
                width = 64
            if merged_t.is_unknown():
                merged_t = TypeRef(width, False, False, None)
            else:
                merged_t.width = width
            op_ref = merged_t
            
            # Fix: Check if the CMP feeds a signed branch (JL, JG, etc.)
            if opcode == 'CMP' and instr.parent and instr.parent.terminator:
                term_op = instr.parent.terminator.opcode.upper()
                if term_op in ['JL', 'JLE', 'JG', 'JGE', 'JS', 'JNS', 'JO', 'JNO']:
                    if op_ref and not op_ref.is_unknown():
                        op_ref.signed = True

        for op in operands:
            if op.memory:
                op.address_refinement = 'ptr'
                idx = operands.index(op)
                if idx < len(src_types):
                    val_type = src_types[idx]
                    if val_type and not val_type.is_unknown():
                        op.inferred_type = val_type.to_string()

        if op_ref and not op_ref.is_unknown():
            instr.op_refinement = op_ref.to_string()

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
        analyzer = RefinementAnalyzer(program)
        analyzer.run()
        legacy_out = ast_to_legacy_program_dict(program, include_instr_locations=False, include_enhancements=True)
        json.dump(legacy_out, sys.stdout, indent=2)
        sys.stdout.write("\n")
    except Exception as e:
        sys.stderr.write(f"Error in Step 8: {repr(e)}\n")
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
