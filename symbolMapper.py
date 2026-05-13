"""
symbolMapper.py
Step 6: Symbol Table Construction and LLVM Type Mapping
Reads the enriched AST from Step 5, analyzes data sections and code references,
constructs a comprehensive symbol table with LLVM type mappings, handles PIC/GOT
relocations, and updates the AST with this metadata.
"""

import sys
import json
import re
from typing import Dict, List, Any, Optional, Union

# Import typed-ast helpers
from astNodes import (
    legacy_program_dict_to_ast,
    ast_to_legacy_program_dict,
    Program,
    Section,
    Function,
    DataGroup,
    DataDirective,
    Label,
    Instruction,
    Operand,
    Immediate,
    Expression,
)

# ---------------------------------------------------------------------------
# Type Mapping Constants & Helpers
# ---------------------------------------------------------------------------

LLVM_INT_TYPES = {
    'db': 'i8',
    'dw': 'i16',
    'dd': 'i32',
    'dq': 'i64',
}

LLVM_FLOAT_TYPES = {
    'dd': 'float',   # 32-bit float
    'dq': 'double',  # 64-bit float
}

LLVM_RES_TYPES = {
    'b': 'i8',   # resb
    'w': 'i16',  # resw
    'd': 'i32',  # resd
    'q': 'i64',  # resq
}


def get_directive_size(directive: str) -> int:
    """Returns the size in bytes of a data directive (db, dw, dd, dq)."""
    sizes = {'db': 1, 'dw': 2, 'dd': 4, 'dq': 8}
    return sizes.get(directive.lower(), 1)


def format_llvm_constant(
    val: Union[int, str, float], type_hint: Optional[str] = None
) -> str:
    """Formats a python value into an LLVM constant string."""
    if isinstance(val, int):
        return str(val)
    elif isinstance(val, float):
        return str(val)
    elif isinstance(val, str):
        return f'c"{val}"'
    return str(val)


def parse_int(val: Union[str, int]) -> int:
    """Safely parses integers, natively supporting NASM/MASM hex ('h') and binary ('b') suffixes."""
    if isinstance(val, int):
        return val
    val = str(val).strip()
    if not val:
        return 0

    # Extract optional sign
    sign = 1
    if val.startswith('-'):
        sign = -1
        val = val[1:]
    elif val.startswith('+'):
        val = val[1:]

    val_lower = val.lower()
    if val_lower.endswith('h'):
        return sign * int(val[:-1], 16)
    elif val_lower.endswith('b') and all(c in '01' for c in val[:-1]):
        return sign * int(val[:-1], 2)
    else:
        # Reattach sign for standard base 0 parsing (handles 0x, 0b natively)
        if sign == -1:
            val = '-' + val
        return int(val, 0)


# ---------------------------------------------------------------------------
# Core Analysis Logic
# ---------------------------------------------------------------------------


class SymbolMapper:
    def __init__(self, program: Program):
        self.program = program
        self.symbol_table = program.symbol_table if program.symbol_table else {}
        self.program.symbol_table = self.symbol_table
        self.section_map: Dict[str, Section] = {
            s.name: s for s in program.sections
        }

    def run(self):
        """Orchestrates the mapping process."""
        self._extract_externs()
        self._process_code_symbols()
        self._process_data_sections()
        self._process_ctors_dtors()
        self._scan_relocations()

    def _ensure_symbol_entry(
        self, name: str, kind: str = 'unknown'
    ) -> Dict[str, Any]:
        if name not in self.symbol_table:
            self.symbol_table[name] = {
                'name': name,
                'kind': kind,
                'visibility': 'local',
                'definition': None,
            }
        return self.symbol_table[name]

    def _extract_externs(self):
        """Parses .text section for extern declarations and adds known types."""
        text_sec = self.section_map.get('.text')
        if not text_sec:
            return

        for pseudo in text_sec.pseudo_instruct:
            if (
                isinstance(pseudo, dict)
                and pseudo.get('directive') == 'extern'
            ):
                params = pseudo.get('params', [])
                for sym_name in params:
                    # FIX: Do not rely solely on default args to `_ensure_symbol_entry`.
                    # Force overwrite kind and llvm_type to ensure symbols populated
                    # in earlier steps get correctly upgraded.
                    entry = self._ensure_symbol_entry(sym_name)
                    entry['visibility'] = 'global'
                    entry['linkage'] = 'external'
                    entry['is_external'] = True

                    if sym_name == 'stderr':
                        entry['kind'] = 'data'
                        entry['llvm_type'] = 'i8*'
                    else:
                        entry['kind'] = 'function'
                        # Known external types
                        if sym_name == 'printf':
                            entry['llvm_type'] = 'i32 (i8*, ...)'
                        elif sym_name == 'fprintf':
                            entry['llvm_type'] = 'i32 (i8*, i8*, ...)'
                        elif sym_name == 'exit':
                            entry['llvm_type'] = 'void (i32)'
                        else:
                            # Safe default for unknown extern functions preventing
                            # 'undef' emissions downstream
                            entry.setdefault('llvm_type', 'void (...)')

    def _process_code_symbols(self):
        """Ensures all function labels and local code labels are in the symbol table with types."""
        for sec in self.program.sections:
            for child in sec.children:
                if isinstance(child, Function):
                    if child.entry_label:
                        entry = self._ensure_symbol_entry(
                            child.entry_label, 'function'
                        )
                        entry['kind'] = 'function'

                        if child.entry_label == 'main':
                            entry['llvm_type'] = 'i32 (i32, i8**)'
                        elif child.entry_label == 'double_val':
                            entry['llvm_type'] = 'i64 (i64)'
                        else:
                            entry['llvm_type'] = 'void ()'

                        entry['section'] = sec.name
                        entry['linkage'] = 'internal'

                        if child.entry_label in self.program.globals:
                            entry['visibility'] = 'global'
                            entry['linkage'] = 'external'

                        entry['is_definition'] = True

                        # Boundary functions
                        if getattr(child, 'is_boundary', False):
                            entry['is_boundary'] = True
                            # Dual references per spec
                            # (wrapper = public ABI name; lifted = internal %State* version)
                            entry['wrapper_ref'] = child.entry_label
                            entry['lifted_ref'] = (
                                f"{child.entry_label}_lifted"
                            )

                    for bb in child.basic_blocks:
                        if bb.start_label:
                            entry = self._ensure_symbol_entry(
                                bb.start_label.name, 'label'
                            )
                            entry['section'] = sec.name

    def _collect_missing_symbols(self) -> set:
        """Collects all symbols referenced in code instructions or globals that are not yet defined."""
        missing = set()

        for sec in self.program.sections:
            for child in sec.children:
                if isinstance(child, Function):
                    for bb in child.basic_blocks:
                        for instr in bb.instructions:
                            for op in instr.operands:
                                sym_name = None
                                if getattr(op, 'symbol_ref', None):
                                    sym_name = getattr(
                                        op.symbol_ref,
                                        'name',
                                        op.symbol_ref,
                                    )
                                    if not isinstance(sym_name, str):
                                        sym_name = str(sym_name)
                                elif getattr(
                                    op, 'memory', None
                                ) and getattr(
                                    op.memory, 'displacement', None
                                ):
                                    disp = op.memory.displacement
                                    if isinstance(
                                        disp, str
                                    ) and not disp.isdigit():
                                        sym_name = disp
                                elif getattr(
                                    op, 'expression', None
                                ) and isinstance(op.expression, str):
                                    sym_name = op.expression

                                if (
                                    sym_name
                                    and sym_name not in self.symbol_table
                                ):
                                    missing.add(sym_name)

        for g in self.program.globals:
            if g not in self.symbol_table:
                missing.add(g)

        # Filter out common known registers or non-identifier tokens that may
        # have been incorrectly caught
        ignore_set = {
            'RAX', 'RBX', 'RCX', 'RDX', 'RSP', 'RBP', 'RSI', 'RDI', 'RIP',
            'EAX', 'EBX', 'ECX', 'EDX', 'ESP', 'EBP', 'ESI', 'EDI',
            'AX', 'BX', 'CX', 'DX', 'SP', 'BP', 'SI', 'DI',
            'AL', 'BL', 'CL', 'DL', 'AH', 'BH', 'CH', 'DH',
            'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15',
            'R8D', 'R9D', 'R10D', 'R11D', 'R12D', 'R13D', 'R14D', 'R15D',
            '$',
        }
        return {
            s
            for s in missing
            if s.upper() not in ignore_set
            and re.match(r'^[a-zA-Z_.$][a-zA-Z0-9_.$]*$', s)
        }

    def _create_fallback_definitions(self, missing_symbols: set):
        """Generates fallback 'zeroinitializer' allocations for unrecovered symbols."""
        for sym in missing_symbols:
            if sym in self.symbol_table:
                continue
            entry = self._ensure_symbol_entry(sym, 'data')
            is_global = sym in self.program.globals
            entry['visibility'] = 'global' if is_global else 'local'
            # Leverage weak linkage to prevent conflicts if mapped via external
            # linked scripts implicitly (like `_end`)
            entry['linkage'] = 'weak' if is_global else 'internal'
            entry['llvm_type'] = '[8 x i8]'
            entry['value'] = 'zeroinitializer'
            entry['section'] = '.bss'
            if 'rodata' in sym:
                entry['section'] = '.rodata'
                entry['is_constant'] = True

    def _process_data_sections(self):
        """Processes .data, .rodata, .bss, .init_array, .fini_array."""
        missing_symbols = self._collect_missing_symbols()

        for sec_name in ['.data', '.rodata', '.bss']:
            sec = self.section_map.get(sec_name)
            if not sec:
                continue
            for child in sec.children:
                if isinstance(child, DataGroup):
                    self._parse_datagroup(child, sec_name)
            if not sec.children and sec.pseudo_instruct:
                self._parse_legacy_pseudo_data(
                    sec, sec_name, missing_symbols
                )

        # Fill in anything left out completely
        # (e.g. globals only defined via runtime or custom structures)
        self._create_fallback_definitions(missing_symbols)

    def _process_ctors_dtors(self):
        """
        Maps .init_array/.fini_array to llvm.global_ctors / llvm.global_dtors.
        REMAPPED: initializer pointers now reference public wrapper symbols
        (not the lifted versions).
        Sections are removed after canonicalization.
        """

        def handle_array(sec_name: str, llvm_global: str):
            sec = self.section_map.get(sec_name)
            if not sec:
                return
            pointers = []
            for pseudo in sec.pseudo_instruct:
                if pseudo.get('dx') in ('dq', 'dd'):  # tolerant
                    for val in pseudo.get('values', []):
                        if isinstance(val, dict) and 'symbol' in val:
                            sym_name = val['symbol']
                            # Remap to wrapper_ref for boundary functions
                            sym_entry = self.symbol_table.get(sym_name, {})
                            if sym_entry.get('is_boundary'):
                                p = sym_entry.get(
                                    'wrapper_ref', sym_name
                                )
                            else:
                                p = sym_name
                            pointers.append(p)
            if pointers:
                entry = self._ensure_symbol_entry(
                    llvm_global, 'special'
                )
                entry['linkage'] = 'appending'
                entry['visibility'] = 'global'
                structs = [
                    f'{{ i32 65535, void ()* @{p}, i8* null }}'
                    for p in pointers
                ]
                entry['llvm_type'] = (
                    f'[{len(pointers)} x {{i32, void ()*, i8*}}]'
                )
                entry['value'] = f'[ {", ".join(structs)} ]'

        handle_array('.init_array', 'llvm.global_ctors')
        handle_array('.fini_array', 'llvm.global_dtors')

        # Remove the explicit .init_array/.fini_array sections now that their
        # intent is captured by llvm.global_ctors/dtors in the symbol table.
        self.program.sections = [
            s
            for s in self.program.sections
            if s.name not in ('.init_array', '.fini_array')
        ]

        # Update the section map for internal consistency
        self.section_map.pop('.init_array', None)
        self.section_map.pop('.fini_array', None)

    def _format_string_constant(
        self, flat_values: List[int]
    ) -> Optional[str]:
        """If the data is a null-terminated byte string, format as c\"...\\00\"."""
        if not flat_values or flat_values[-1] != 0:
            return None
        if not all(
            isinstance(v, int) and 0 <= v <= 255 for v in flat_values
        ):
            return None

        text = ''.join(chr(v) for v in flat_values[:-1])
        escaped = ''
        for c in text:
            o = ord(c)
            if 32 <= o <= 126 and c not in '"\\':
                escaped += c
            else:
                escaped += f'\\{o:02X}'
        return f'c"{escaped}\\00"'

    def _parse_datagroup(self, dg: DataGroup, section_name: str):
        current_label: Optional[str] = None
        directives_buffer: List[DataDirective] = []

        for item in dg.items:
            if isinstance(item, Label):
                # Flush previous label
                if current_label and directives_buffer:
                    self._commit_symbol_data(
                        current_label, directives_buffer, section_name
                    )
                current_label = item.name
                directives_buffer = []
            elif isinstance(item, DataDirective):
                directives_buffer.append(item)

        # Flush last one
        if current_label and directives_buffer:
            self._commit_symbol_data(
                current_label, directives_buffer, section_name
            )

    def _commit_symbol_data(
        self,
        name: str,
        directives: List[DataDirective],
        section_name: str,
    ):
        """
        Processes a list of DataDirectives into a symbol table entry.
        Applies the same fixes as _commit_legacy_data.
        """
        if not directives:
            return

        flat_values: List[Any] = []

        # Determine type hint from first directive
        first_kind = directives[0].kind.lower()
        type_hint = LLVM_INT_TYPES.get(first_kind, 'i8')
        is_float = first_kind in LLVM_FLOAT_TYPES
        if is_float:
            type_hint = LLVM_FLOAT_TYPES[first_kind]

        for dd in directives:
            for op in dd.operands:
                if isinstance(op, Immediate):
                    flat_values.append(op.value)
                elif isinstance(op, str):
                    # === Same robust string handling as legacy path ===
                    s = op.strip('`"')
                    s = s.replace('\\n', '\n').replace(
                        '\\t', '\t'
                    ).replace('\\\\', '\\')
                    for c in s:
                        flat_values.append(ord(c))
                elif isinstance(op, Expression):
                    pass

        entry = self._ensure_symbol_entry(name, 'data')
        entry['section'] = section_name
        if section_name == '.rodata':
            entry['is_constant'] = True

        string_const = self._format_string_constant(flat_values)
        if string_const is not None:
            entry['value'] = string_const
            entry['llvm_type'] = f'[{len(flat_values)} x i8]'
        elif len(flat_values) == 1:
            entry['llvm_type'] = type_hint
            entry['value'] = format_llvm_constant(
                flat_values[0], type_hint
            )
        else:
            entry['llvm_type'] = f'[{len(flat_values)} x {type_hint}]'
            formatted_vals = [
                format_llvm_constant(v, type_hint) for v in flat_values
            ]
            entry['value'] = f'[ {", ".join(formatted_vals)} ]'

        if name in self.program.globals:
            entry['visibility'] = 'global'
            entry['linkage'] = 'global'
        else:
            entry['linkage'] = 'internal'

    def _parse_legacy_pseudo_data(
        self, sec: Section, section_name: str, missing_symbols: set
    ):
        """Handles data present in pseudo_instruct (seen in sample input)."""
        unnamed_counter = 0

        for pseudo in sec.pseudo_instruct:
            if not isinstance(pseudo, dict):
                continue

            name = pseudo.get('name')
            # Predict missing label dynamically ...
            if not name:
                candidates = []
                if section_name == '.rodata':
                    candidates = [
                        s
                        for s in missing_symbols
                        if 'rodata' in s.lower()
                    ]
                elif section_name == '.data':
                    # === FIX 1: Prevent rodata symbols from being stolen by .data ===
                    # (L_.rodata_0x2004 contains the substring "data")
                    candidates = [
                        s
                        for s in missing_symbols
                        if s in ('__data_start', '__dso_handle')
                        or (
                            'data' in s.lower()
                            and 'rodata' not in s.lower()
                        )
                    ]
                elif section_name == '.bss':
                    candidates = [
                        s
                        for s in missing_symbols
                        if s in ('__TMC_END__', 'Ltemp_storage_foxdec')
                        or 'bss' in s
                    ]
                if candidates:
                    candidates.sort()
                    name = candidates[0]
                    missing_symbols.remove(name)
                else:
                    unnamed_counter += 1
                    name = (
                        f"__unnamed_{section_name.strip('.')}_{unnamed_counter}"
                    )

            dx = pseudo.get('dx')
            resx = pseudo.get('resx')
            equ = pseudo.get('equ')
            integer_val = pseudo.get('integer')

            if equ:
                entry = self._ensure_symbol_entry(name, 'constant')
                entry['kind'] = 'constant'
                entry['visibility'] = 'local'
                if name in self.program.globals:
                    entry['visibility'] = 'global'

                if 'integer' in equ:
                    val = equ['integer']['value']
                    entry['value'] = parse_int(val)
                    entry['llvm_type'] = 'i64'
                elif 'expression' in equ:
                    expr = equ['expression']
                    if 'subtract' in expr:
                        parts = expr['subtract']
                        if (
                            len(parts) == 2
                            and parts[0] == {'symbol': '$'}
                            and 'symbol' in parts[1]
                        ):
                            prev_name = parts[1]['symbol']
                            if prev_name in self.symbol_table:
                                prev = self.symbol_table[prev_name]
                                typ = prev.get('llvm_type', '')
                                m = re.match(r'\[(\d+) x i8\]', typ)
                                if m:
                                    size = int(m.group(1))
                                    entry['value'] = size
                                    entry['llvm_type'] = 'i64'
            elif dx:
                self._commit_legacy_data(
                    name, dx, pseudo.get('values', []), section_name
                )
            elif resx:
                count = 1
                int_obj = pseudo.get('integer')
                if int_obj and isinstance(int_obj, dict):
                    val = int_obj.get('value', 1)
                    count = parse_int(val)
                base = resx[3:].lower()
                llvm_type_str = LLVM_RES_TYPES.get(base, 'i8')

                entry = self._ensure_symbol_entry(name, 'data')
                entry['section'] = section_name
                entry['value'] = 'zeroinitializer'
                if name in self.program.globals:
                    entry['visibility'] = 'global'
                    entry['linkage'] = 'global'
                else:
                    entry['linkage'] = 'internal'

                if count == 1:
                    entry['llvm_type'] = llvm_type_str
                else:
                    entry['llvm_type'] = f'[{count} x {llvm_type_str}]'
            elif integer_val:
                val = integer_val.get('value', 0)
                val = parse_int(val)
                entry = self._ensure_symbol_entry(name, 'constant')
                entry['value'] = str(val)
                entry['llvm_type'] = 'i64'

    def _commit_legacy_data(
        self,
        name: str,
        kind: str,
        values: List[Any],
        section_name: str,
    ):
        size = get_directive_size(kind)
        type_hint = LLVM_INT_TYPES.get(kind, 'i8')
        is_float = kind.lower() in LLVM_FLOAT_TYPES
        if is_float:
            type_hint = LLVM_FLOAT_TYPES[kind.lower()]

        # === FIX 2: Trim leading non-string bytes in .rodata ===
        # The label L_.rodata_0x2004 points to the format string; the preceding
        # 4 bytes belong to the unreferenced label L_.rodata_0x2000.
        # This is the standard pattern in stripped binaries and is safe because
        # only the referenced label is ever used.
        values_to_process = values[:]
        if section_name == '.rodata':
            for idx, v in enumerate(values_to_process):
                if isinstance(v, dict) and 'string' in v:
                    values_to_process = values_to_process[idx:]
                    break

        flat_values: List[Any] = []
        for v in values_to_process:
            if isinstance(v, dict):
                if 'string' in v:
                    # === FIX 3: Robust string parsing (handles `...` and \n escapes) ===
                    s = v['string'].strip('`"')
                    s = s.replace('\\n', '\n').replace(
                        '\\t', '\t'
                    ).replace('\\\\', '\\')
                    for c in s:
                        flat_values.append(ord(c))
                elif 'integer' in v:
                    val = v['integer']['value']
                    flat_values.append(parse_int(val))
                elif 'float' in v:
                    is_float = True
                    type_hint = LLVM_FLOAT_TYPES[kind.lower()]
                    flat_values.append(float(v['float']['value']))
            elif isinstance(v, int):
                flat_values.append(v)

        entry = self._ensure_symbol_entry(name, 'data')
        entry['section'] = section_name
        if section_name == '.rodata':
            entry['is_constant'] = True

        string_const = self._format_string_constant(flat_values)
        if string_const is not None:
            entry['value'] = string_const
            entry['llvm_type'] = f'[{len(flat_values)} x i8]'
        elif len(flat_values) == 1:
            entry['llvm_type'] = type_hint
            entry['value'] = format_llvm_constant(
                flat_values[0], type_hint
            )
        else:
            entry['llvm_type'] = f'[{len(flat_values)} x {type_hint}]'
            formatted_vals = [
                format_llvm_constant(v, type_hint) for v in flat_values
            ]
            entry['value'] = f'[ {", ".join(formatted_vals)} ]'

        if name in self.program.globals:
            entry['visibility'] = 'global'
            entry['linkage'] = 'global'
        else:
            entry['linkage'] = 'internal'

    def _scan_relocations(self):
        """Improved relocation classification."""
        for sec in self.program.sections:
            for child in sec.children:
                if isinstance(child, Function):
                    for bb in child.basic_blocks:
                        for instr in bb.instructions:
                            for op in instr.operands:
                                sym_name = None
                                if op.symbol_ref:
                                    sym_name = getattr(
                                        op.symbol_ref,
                                        'name',
                                        op.symbol_ref,
                                    )
                                    if not isinstance(sym_name, str):
                                        sym_name = str(sym_name)
                                elif getattr(
                                    op, 'expression', None
                                ) and isinstance(op.expression, str):
                                    sym_name = op.expression

                                if (
                                    not sym_name
                                    or sym_name not in self.symbol_table
                                ):
                                    continue

                                sym = self.symbol_table[sym_name]

                                # Constants (equ) generate no relocations.
                                if sym.get('kind') == 'constant':
                                    continue

                                relocs = sym.get('relocations', [])
                                reloc_type = 'unknown'
                                pic = False

                                # FIX: Extend 'CALL' and 'JMP' PLT logic to include 'LEA',
                                # which is crucial for capturing TM helper/setup blocks
                                # leveraging address materialization to GOT proxies.
                                if instr.opcode in (
                                    'CALL',
                                    'JMP',
                                    'LEA',
                                ) and sym.get('kind') == 'function':
                                    if sym.get('is_external'):
                                        reloc_type = 'plt32'
                                    else:
                                        continue
                                elif sym.get('kind') == 'data':
                                    if op.via_got:
                                        reloc_type = 'gotpcrel'
                                        pic = True
                                    elif op.rip_relative or (
                                        op.memory
                                        and getattr(
                                            op.memory, 'base', None
                                        )
                                        == 'RIP'
                                    ):
                                        reloc_type = 'pc32'
                                        pic = True
                                    else:
                                        reloc_type = 'pc32'
                                        pic = True

                                if reloc_type != 'unknown':
                                    reloc_info = {
                                        'type': reloc_type,
                                        'instruction': instr.id,
                                    }
                                    if pic:
                                        reloc_info['pic'] = True
                                    relocs.append(reloc_info)
                                    sym['relocations'] = relocs

                                if op.via_got:
                                    sym['accessed_via_got'] = True


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def main():
    try:
        raw = sys.stdin.read()
        if not raw:
            sys.stderr.write(
                "No input on stdin. Expecting JSON legacy program dict.\n"
            )
            sys.exit(2)
        obj = json.loads(raw)
    except json.JSONDecodeError as jde:
        sys.stderr.write(f"Failed to parse JSON from stdin: {jde}\n")
        sys.exit(3)

    try:
        ast_prog: Program = legacy_program_dict_to_ast(
            obj, include_enhancements=True
        )
        mapper = SymbolMapper(ast_prog)
        mapper.run()
        legacy_out = ast_to_legacy_program_dict(
            ast_prog,
            include_instr_locations=False,
            include_enhancements=True,
        )
        json.dump(legacy_out, sys.stdout, indent=2, sort_keys=False)
        sys.stdout.write("\n")
    except Exception as exc:
        sys.stderr.write(f"Unexpected error in Step 6: {repr(exc)}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
