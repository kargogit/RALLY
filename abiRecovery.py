"""
abiRecovery.py
Step 7: Intraprocedural ABI Recovery (Arguments, Return Type, Stack Layout)
Builds on the enriched AST from previous steps. For each internal function:
- Detects standard frame-pointer prolog to determine uses_frame_pointer and abi_compliance.
- Recovers register-passed arguments using System V x86-64 ABI order, conservatively confirmed
  by early reads before any write in a linear scan (approximating execution order).
- If frame pointer is used, additionally recovers stack-passed arguments from positive RBP offsets.
- Infers return type based on writes to RAX/EAX* (i64) or lack thereof (void), with special
  handling for 'main'.
- Recovers local stack slots from negative RBP offsets (only when frame pointer is used).
- Annotates the Function node and updates the SymbolTable with the recovered signature.
"""
import sys
import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple

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
    None: 8,  # default to 8-byte when no explicit size
}

def _get_writes_reads(instr: Instruction) -> Tuple[Set[str], Set[str]]:
    writes: Set[str] = set()
    reads: Set[str] = set()

    # Rough destination detection: first operand for most arithmetic/mov ops
    if instr.operands:
        dest_op = instr.operands[0]
        if dest_op.register:
            writes.add(dest_op.register)
        # Memory destinations read base/index
        if dest_op.memory:
            if dest_op.memory.base:
                reads.add(dest_op.memory.base)
            if dest_op.memory.index:
                reads.add(dest_op.memory.index)

    # Heuristic: Instructions where the first operand (destination) is ALSO a source (read).
    # Most arithmetic/logic/shift instructions read the destination value to modify it.
    # Stack (PUSH) and Control (CALL) instructions also read the first operand.
    # Move/Load instructions (MOV, LEA, POP) do NOT read the destination; they purely define it.
    READS_FIRST_OPERAND = {
        # Arithmetic / Logic
        "ADD", "ADC", "SUB", "SBB", "AND", "OR", "XOR", "TEST", "CMP",
        # Shifts / Rotates
        "SHL", "SAL", "SAR", "SHR", "ROL", "ROR", "RCL", "RCR", "SHLD", "SHRD",
        # Unary
        "INC", "DEC", "NEG", "NOT",
        # Mul / Div
        "IMUL", "IDIV", "DIV", "MUL",
        # Conditional Moves (read dest if condition is met)
        "CMOVE", "CMOVNE", "CMOVG", "CMOVGE", "CMOVL", "CMOVLE", "CMOVA", "CMOVAE",
        "CMOVB", "CMOVBE", "CMOVS", "CMOVNS", "CMOVP", "CMOVNP",
        # Stack / Control
        "PUSH", "CALL",
        # Exchange
        "XCHG", "XADD", "CMPXCHG"
    }

    reads_first = instr.opcode in READS_FIRST_OPERAND

    # Sources (all operands)
    for i, op in enumerate(instr.operands):
        if op.register:
            # If this is the first operand (destination) and the instruction does NOT
            # read the destination, skip adding it to reads.
            if i == 0 and not reads_first:
                continue
            reads.add(op.register)
            
        if op.memory:
            if op.memory.base:
                reads.add(op.memory.base)
            if op.memory.index:
                reads.add(op.memory.index)

    # Special cases
    if instr.opcode == "PUSH":
        writes.add("RSP")
    if instr.opcode == "POP":
        reads.add("RSP")
        # POP writes to the operand (already handled by dest logic), but it does NOT read the operand value
        # The logic above (i==0 and not reads_first) ensures we don't incorrectly add dest to reads.
        if instr.operands and instr.operands[0].register:
            writes.add(instr.operands[0].register)
    if instr.opcode in ("DIV", "IDIV"):
        reads.update({"RAX", "RDX"})

    return writes, reads

# ---------------------------------------------------------------------------
# Core Recovery Logic
# ---------------------------------------------------------------------------
class AbiRecoverer:
    def __init__(self, program: Program):
        self.program = program
        self.symbol_table = program.symbol_table

    def run(self):
        """Process all internal functions."""
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

        for bb in func.basic_blocks:  # ordered list approximates discovery/execution order
            for instr in bb.instructions:
                # Register early-use for arguments
                writes, reads = _get_writes_reads(instr)
                for reg in reads:
                    if reg in GP_ARG_REGS and reg not in defined_regs:
                        early_int_arg_regs.add(reg)
                defined_regs.update(writes)

                # Memory accesses for stack args / locals
                for op in instr.operands:
                    if op.memory and op.memory.base in ("RBP", "RSP") and isinstance(op.memory.displacement, int):
                        mem_accesses.append((op.memory.displacement, op, instr))

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
                        first_use=None,  # could record location if desired
                    )
                )

        # ----- Recover stack-passed arguments (only with frame pointer) -----
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
                    # Non-sequential or unexpected → stop promoting stack args
                    break

        # ----- Recover local stack slots (only with frame pointer, negative offsets) -----
        stack_slots: List[StackSlot] = []
        if uses_fp:
            offset_to_sizes = defaultdict(set)
            for disp, op, _ in mem_accesses:
                if disp < 0:
                    size_str = op.size
                    size = SIZE_MAP.get(size_str, 8)
                    offset_to_sizes[disp].add(size)

            for off in sorted(offset_to_sizes.keys()):
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
                        index=len(stack_slots),
                    )
                )

        # ----- Infer return type -----
        has_rax_write = any(
            len(i.operands) > 0 and i.operands[0].register in RETURN_INT_REGS
            for bb in func.basic_blocks
            for i in bb.instructions
        )

        if func.entry_label == "main":
            return_type = "i32"
            # Force canonical main signature
            arguments = [
                ArgumentDescriptor(kind="register", location="RDI", index=0, inferred_type="i32"),
                ArgumentDescriptor(kind="register", location="RSI", index=1, inferred_type="i8**"),
            ]
        elif func.noreturn_kind:
            return_type = "void"
        else:
            return_type = "i64" if has_rax_write else "void"

        # ----- Set ABI compliance -----
        if func.entry_label == "main":
            abi_compliance = "standard"
        elif uses_fp and len([a for a in arguments if a.kind == "stack"]) == 0:
            abi_compliance = "standard"
        elif arguments:
            abi_compliance = "partial"
        elif stack_slots:
            abi_compliance = "custom"
        else:
            abi_compliance = "raw"

        # ----- Annotate function -----
        func.arguments = arguments
        func.stack_slots = stack_slots
        func.return_type = return_type
        func.abi_compliance = abi_compliance

        # ----- Update symbol table with recovered signature -----
        if func.entry_label in self.symbol_table:
            sym = self.symbol_table[func.entry_label]
            arg_types = []
            for ad in arguments:
                typ = ad.inferred_type or "i64"
                if ad.kind == "float":
                    typ = "double"
                arg_types.append(typ)
            args_str = ", ".join(arg_types) if arg_types else ""
            ret = return_type or "void"
            sym["llvm_type"] = f"{ret} ({args_str})" if args_str else f"{ret} ()"

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
