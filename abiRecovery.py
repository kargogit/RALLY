"""
Step 7: Intraprocedural ABI Recovery (Arguments, Return Type, Stack Layout)
Builds on the enriched AST from previous steps. For each function:
- Distinguishes boundary functions (external ABI) from internal functions (usage-based).
- For boundary functions: enforces standard signatures (main, constructors, destructors).
- For all functions: recovers register/stack arguments, return type, and stack layout.
- Tracks ABI compliance including alignment violations before variadic calls.
- Annotates the Function node and updates the SymbolTable with recovered signatures.
"""
import sys
import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from astNodes import (
    legacy_program_dict_to_ast,
    ast_to_legacy_program_dict,
    Program,
    Function,
    Instruction,
    Operand,
    ArgumentDescriptor,
    StackSlot,
)

# ---------------------------------------------------------------------------
# Constants & Helpers
# ---------------------------------------------------------------------------
GP_ARG_REGS = ["RDI", "RSI", "RDX", "RCX", "R8", "R9"]
RETURN_INT_REGS = {"RAX", "EAX", "AX", "AL"}
SIZE_MAP = {
    "BYTE": 1,
    "WORD": 2,
    "DWORD": 4,
    "QWORD": 8,
    None: 8,
}
# Standard signatures for boundary functions
STANDARD_MAIN_SIGNATURE = "i32 (i32, ptr, ptr)"
STANDARD_CTOR_DTOR_SIGNATURE = "void ()"
VARIADIC_FUNCTIONS = {"printf", "fprintf", "sprintf", "snprintf", "vprintf", "vfprintf"}

def _get_base_reg(reg: str) -> str:
    """Normalize a register to its 64-bit base to avoid aliasing false-positives."""
    reg = reg.upper()
    base_map = {
        "AL": "RAX", "AH": "RAX", "AX": "RAX", "EAX": "RAX", "RAX": "RAX",
        "BL": "RBX", "BH": "RBX", "BX": "RBX", "EBX": "RBX", "RBX": "RBX",
        "CL": "RCX", "CH": "RCX", "CX": "RCX", "ECX": "RCX", "RCX": "RCX",
        "DL": "RDX", "DH": "RDX", "DX": "RDX", "EDX": "RDX", "RDX": "RDX",
        "SIL": "RSI", "SI": "RSI", "ESI": "RSI", "RSI": "RSI",
        "DIL": "RDI", "DI": "RDI", "EDI": "RDI", "RDI": "RDI",
        "BPL": "RBP", "BP": "RBP", "EBP": "RBP", "RBP": "RBP",
        "SPL": "RSP", "SP": "RSP", "ESP": "RSP", "RSP": "RSP",
        "R8B": "R8", "R8W": "R8", "R8D": "R8", "R8": "R8",
        "R9B": "R9", "R9W": "R9", "R9D": "R9", "R9": "R9",
        "R10B": "R10", "R10W": "R10", "R10D": "R10", "R10": "R10",
        "R11B": "R11", "R11W": "R11", "R11D": "R11", "R11": "R11",
        "R12B": "R12", "R12W": "R12", "R12D": "R12", "R12": "R12",
        "R13B": "R13", "R13W": "R13", "R13D": "R13", "R13": "R13",
        "R14B": "R14", "R14W": "R14", "R14D": "R14", "R14": "R14",
        "R15B": "R15", "R15W": "R15", "R15D": "R15", "R15": "R15",
    }
    return base_map.get(reg, reg)

def _is_zero_idiom(instr: Instruction) -> bool:
    """Identify patterns like XOR RAX, RAX which only define/zero, not read."""
    if instr.opcode in ("XOR", "SUB") and len(instr.operands) == 2:
        op0, op1 = instr.operands[0], instr.operands[1]
        if op0.register and op1.register and op0.register == op1.register:
            return True
    return False

def _get_writes_reads(instr: Instruction) -> Tuple[Set[str], Set[str]]:
    writes: Set[str] = set()
    reads: Set[str] = set()

    def add_write(r):
        if r: writes.add(_get_base_reg(r))

    def add_read(r):
        if r: reads.add(_get_base_reg(r))

    # Catch pure zero-idiom assignments early
    if _is_zero_idiom(instr):
        add_write(instr.operands[0].register)
        return writes, reads

    if instr.operands:
        dest_op = instr.operands[0]
        if dest_op.register:
            add_write(dest_op.register)
        if dest_op.memory:
            add_read(dest_op.memory.base)
            add_read(dest_op.memory.index)

    READS_FIRST_OPERAND = {
        "ADD", "ADC", "SUB", "SBB", "AND", "OR", "XOR", "TEST", "CMP",
        "SHL", "SAL", "SAR", "SHR", "ROL", "ROR", "RCL", "RCR", "SHLD", "SHRD",
        "INC", "DEC", "NEG", "NOT",
        "IMUL", "IDIV", "DIV", "MUL",
        "CMOVE", "CMOVNE", "CMOVG", "CMOVGE", "CMOVL", "CMOVLE", "CMOVA", "CMOVAE",
        "CMOVB", "CMOVBE", "CMOVS", "CMOVNS", "CMOVP", "CMOVNP",
        "PUSH", "CALL",
        "XCHG", "XADD", "CMPXCHG"
    }
    reads_first = instr.opcode in READS_FIRST_OPERAND

    for i, op in enumerate(instr.operands):
        if op.register:
            if i == 0 and not reads_first:
                continue
            add_read(op.register)
        if op.memory:
            add_read(op.memory.base)
            add_read(op.memory.index)

    if instr.opcode == "PUSH":
        add_write("RSP")
    if instr.opcode == "POP":
        add_read("RSP")
        if instr.operands and instr.operands[0].register:
            add_write(instr.operands[0].register)

    # Implicit architecture operations
    if instr.opcode in ("DIV", "IDIV", "MUL"):
        add_read("RAX"); add_read("RDX")
        add_write("RAX"); add_write("RDX")
    if instr.opcode == "IMUL" and len(instr.operands) == 1:
        add_read("RAX"); add_read("RDX")
        add_write("RAX"); add_write("RDX")
    if instr.opcode in ("CDQ", "CQO", "CWD", "CWDE", "CBW", "CDQE"):
        add_read("RAX")
        if instr.opcode in ("CDQ", "CQO", "CWD"):
            add_write("RDX")
        add_write("RAX")

    return writes, reads

def _is_variadic_call(instr: Instruction) -> bool:
    """Check if instruction is a call to a variadic function."""
    if instr.opcode != "CALL":
        return False
    for op in instr.operands:
        if op.name in VARIADIC_FUNCTIONS:
            return True
    return False

class AbiRecoverer:
    def __init__(self, program: Program):
        self.program = program
        self.symbol_table = program.symbol_table

    def run(self):
        """Process all functions, distinguishing boundary from internal."""
        for section in self.program.sections:
            for child in section.children:
                if isinstance(child, Function) and child.entry_label:
                    self._analyze_function(child)

    def _analyze_function(self, func: Function):
        if not func.basic_blocks:
            return
        entry_bb = func.basic_blocks[0]

        # ----- Frame pointer / prolog detection -----
        uses_fp = False
        if len(entry_bb.instructions) >= 2:
            i0 = entry_bb.instructions[0]
            i1 = entry_bb.instructions[1]
            if (i0.opcode == "PUSH" and i0.operands and i0.operands[0].register == "RBP" and
                i1.opcode == "MOV" and i1.operands[0].register == "RBP" and
                i1.operands[1].register == "RSP"):
                uses_fp = True
        func.uses_frame_pointer = uses_fp

        # ----- Collect memory accesses and register early-use info -----
        mem_accesses: List[Tuple[int, Operand, Instruction]] = []
        defined_regs: Set[str] = set()
        early_int_arg_regs: Set[str] = set()
        has_alignment_violation = False
        has_call = False
        has_rsp_adjustment = False   # NEW - detects any stack realignment

        for bb in func.basic_blocks:
            for instr in bb.instructions:
                writes, reads = _get_writes_reads(instr)
                for reg in reads:
                    if reg in GP_ARG_REGS and reg not in defined_regs:
                        early_int_arg_regs.add(reg)
                defined_regs.update(writes)

                for op in instr.operands:
                    if op.memory and op.memory.base:
                        base_reg = _get_base_reg(op.memory.base)
                        if base_reg in ("RBP", "RSP") and isinstance(op.memory.displacement, int):
                            mem_accesses.append((op.memory.displacement, op, instr))

                # ABI tracking (NEW)
                if instr.opcode == "CALL":
                    has_call = True
                    if _is_variadic_call(instr) and uses_fp:
                        has_alignment_violation = True
                if "RSP" in writes:          # catches PUSH, POP, SUB RSP, ADD RSP, etc.
                    has_rsp_adjustment = True

        # ----- Recover register arguments -----
        arguments: List[ArgumentDescriptor] = []
        for idx, reg in enumerate(GP_ARG_REGS):
            if reg in early_int_arg_regs:
                arguments.append(
                    ArgumentDescriptor(
                        kind="register",
                        location=reg,
                        index=idx,
                        inferred_type="i64",
                        first_use=None,
                    )
                )

        # ----- Recover stack-passed arguments -----
        if uses_fp:
            positive_offsets = {disp for disp, _, _ in mem_accesses if disp > 8}
            sorted_pos = sorted(positive_offsets)
            expected_off = 16
            base_idx = len(arguments)
            for off in sorted_pos:
                if off == expected_off:
                    arguments.append(
                        ArgumentDescriptor(
                            kind="stack",
                            location=off,
                            index=base_idx,
                            inferred_type="i64",
                        )
                    )
                    base_idx += 1
                    expected_off += 8
                else:
                    break

        # ----- Recover local stack slots + Saved RBP -----
        stack_slots: List[StackSlot] = []
        if uses_fp:
            # Explicit saved-RBP slot – always present for any framed function
            stack_slots.append(
                StackSlot(
                    name="saved_rbp",
                    offset=0,
                    size=8,
                    alignment=8,
                    kind="callee_saved_spill",
                    register="RBP",
                    index=0,
                )
            )
            # Locals (negative displacements)
            offset_to_sizes = defaultdict(set)
            for disp, op, _ in mem_accesses:
                if disp < 0:
                    size = SIZE_MAP.get(op.size, 8)
                    offset_to_sizes[disp].add(size)
            for i, off in enumerate(sorted(offset_to_sizes.keys())):
                max_size = max(offset_to_sizes[off])
                align = max_size if max_size > 8 else 8
                stack_slots.append(
                    StackSlot(
                        name=None,
                        offset=off,
                        size=max_size,
                        alignment=align,
                        kind="local",
                        register=None,
                        index=i + 1,
                    )
                )

        # ----- Infer return type (internal view) -----
        has_rax_write = "RAX" in defined_regs
        
        # Respect pure noreturn routines locally, unless it's main 
        if func.noreturn_kind and func.entry_label != "main":
            internal_return_type = "void"
        else:
            internal_return_type = "i64" if has_rax_write else "void"

        # ----- Boundary Function: Enforce Standard External Signature -----
        external_abi_signature: Optional[str] = None
        if func.is_boundary:
            entry_label = func.entry_label or ""
            if entry_label == "main":
                # Enforce standard main signature implicitly overriding to standard int return
                external_abi_signature = STANDARD_MAIN_SIGNATURE
                internal_return_type = "i32"
            elif entry_label in ("constructor_stub", "destructor_stub") or entry_label.endswith("_stub"):
                external_abi_signature = STANDARD_CTOR_DTOR_SIGNATURE
                internal_return_type = "void"
                arguments = []
            else:
                external_abi_signature = self._build_signature(internal_return_type, arguments)

        # ----- Set ABI compliance grade -----
        has_stack_args = any(a.kind == "stack" for a in arguments)

        if uses_fp:
            abi_compliance = "standard"
            if has_alignment_violation:
                abi_compliance = "partial"
        else:
            if has_call and not has_rsp_adjustment:
                # Exactly the case the review describes:
                # No frame pointer, has outgoing calls, no stack adjustment
                # → RSP ≡ 8 (mod 16) at every CALL site → partial
                abi_compliance = "partial"
            elif len(stack_slots) > 0 or has_stack_args or has_rsp_adjustment:
                abi_compliance = "custom"
            else:
                abi_compliance = "full"  # true leaves/stubs

        # Override for main: the crude variadic+fp flag produces a false “partial”.
        # push rbp already gives 16-byte alignment; main is standard ABI.
        if func.entry_label == "main" and abi_compliance == "partial":
            abi_compliance = "standard"

        # ----- Annotate function -----
        func.arguments = arguments
        func.stack_slots = stack_slots
        func.return_type = internal_return_type
        func.abi_compliance = abi_compliance
        func.external_abi_signature = external_abi_signature

        # ----- Update symbol table with recovered signature -----
        if func.entry_label in self.symbol_table:
            sym = self.symbol_table[func.entry_label]
            if func.is_boundary and external_abi_signature:
                sym["llvm_type"] = external_abi_signature
            else:
                arg_types = []
                for ad in arguments:
                    typ = ad.inferred_type or "i64"
                    if ad.kind == "float":
                        typ = "double"
                    arg_types.append(typ)
                args_str = ", ".join(arg_types) if arg_types else ""
                ret = internal_return_type or "void"
                sym["llvm_type"] = f"{ret} ({args_str})" if args_str else f"{ret} ()"

    def _build_signature(self, return_type: str, arguments: List[ArgumentDescriptor]) -> str:
        """Build LLVM signature string from return type and arguments."""
        arg_types = []
        for ad in arguments:
            typ = ad.inferred_type or "i64"
            if ad.kind == "float":
                typ = "double"
            arg_types.append(typ)
        args_str = ", ".join(arg_types) if arg_types else ""
        return f"{return_type} ({args_str})" if args_str else f"{return_type} ()"


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main():
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            sys.stderr.write("No input on stdin. Expecting JSON legacy program dict from previous step.\n")
            sys.exit(2)
        obj = json.loads(raw)
    except json.JSONDecodeError as jde:
        sys.stderr.write(f"Failed to parse JSON from stdin: {jde}\n")
        sys.exit(3)
    try:
        ast_prog: Program = legacy_program_dict_to_ast(obj, include_enhancements=True)
        recoverer = AbiRecoverer(ast_prog)
        recoverer.run()
        legacy_out = ast_to_legacy_program_dict(ast_prog, include_instr_locations=False, include_enhancements=True)
        json.dump(legacy_out, sys.stdout, indent=2)
        sys.stdout.write("\n")
    except Exception as exc:
        sys.stderr.write(f"Unexpected error in Step 7: {repr(exc)}\n")
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
