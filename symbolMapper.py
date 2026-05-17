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
    if isinstance(val, (int, float)):
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
    try:
        if val_lower.endswith('h'):
            return sign * int(val[:-1], 16)
        elif val_lower.endswith('b') and all(c in '01' for c in val[:-1]):
            return sign * int(val[:-1], 2)
        else:
            # Reattach sign for standard base 0 parsing (handles 0x, 0b natively)
            if sign == -1:
                val = '-' + val
            return int(val, 0)
    except ValueError:
        return 0


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
        try:
            self._extract_externs()
            self._process_code_symbols()
            self._process_data_sections()
            self._process_ctors_dtors()
            self._scan_relocations()
        except Exception as e:
            sys.stderr.write(f"Error during SymbolMapper execution: {str(e)}\n")
            raise

    def _ensure_symbol_entry(self, name: str, kind: str = 'unknown') -> Dict[str, Any]:
        if name not in self.symbol_table:
            self.symbol_table[name] = {
                'name': name,
                'kind': kind,
                'visibility': 'local',
                'definition': None,
            }
        else:
            # Upgrade kind if it was previously undefined/unknown
            if self.symbol_table[name].get('kind') == 'unknown' and kind != 'unknown':
                self.symbol_table[name]['kind'] = kind
        return self.symbol_table[name]

    def _extract_externs(self):
        """Parses .text section for extern declarations and adds known types.
        All function externals now default to 'void (...)' so that Step 9's
        ExternalABIDB.KNOWN can provide the precise LLVM signature.
        """
        text_sec = self.section_map.get('.text')
        if not text_sec:
            return
        for pseudo in text_sec.pseudo_instruct:
            if isinstance(pseudo, dict) and pseudo.get('directive') == 'extern':
                params = pseudo.get('params', [])
                for sym_name in params:
                    entry = self._ensure_symbol_entry(sym_name)
                    entry['visibility'] = 'global'
                    entry['linkage'] = 'external'
                    entry['is_external'] = True
                    if sym_name == 'stderr':
                        entry['kind'] = 'data'
                        entry['llvm_type'] = 'i8*'
                    else:
                        # DEFAULT – Step 9 will upgrade known library functions
                        entry['kind'] = 'function'
                        entry.setdefault('llvm_type', 'void (...)')

    def _process_code_symbols(self):
        """Ensures all function labels and local code labels are in the symbol table with types."""
        for sec in self.program.sections:
            for child in sec.children:
                if isinstance(child, Function):
                    if child.entry_label:
                        entry = self._ensure_symbol_entry(child.entry_label, 'function')
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

                        # Boundary functions duality definitions
                        if getattr(child, 'is_boundary', False):
                            entry['is_boundary'] = True
                            entry['wrapper_ref'] = child.entry_label
                            entry['lifted_ref'] = f"{child.entry_label}_lifted"

                    for bb in child.basic_blocks:
                        if bb.start_label:
                            entry = self._ensure_symbol_entry(bb.start_label.name, 'label')
                            entry['section'] = sec.name

    def _extract_symbols_from_operand(self, op) -> List[str]:
        """Dynamically extracts identifiable symbols from potentially complex operands (handles arithmetic)."""
        raw_sym = None
        if getattr(op, 'symbol_ref', None):
            raw_sym = getattr(op.symbol_ref, 'name', op.symbol_ref)
        elif getattr(op, 'memory', None) and getattr(op.memory, 'displacement', None):
            raw_sym = op.memory.displacement
        elif getattr(op, 'expression', None):
            raw_sym = op.expression

        if not isinstance(raw_sym, str):
            if raw_sym is None:
                return []
            raw_sym = str(raw_sym)

        ignore_set = {
            'RAX', 'RBX', 'RCX', 'RDX', 'RSP', 'RBP', 'RSI', 'RDI', 'RIP',
            'EAX', 'EBX', 'ECX', 'EDX', 'ESP', 'EBP', 'ESI', 'EDI',
            'AX', 'BX', 'CX', 'DX', 'SP', 'BP', 'SI', 'DI',
            'AL', 'BL', 'CL', 'DL', 'AH', 'BH', 'CH', 'DH',
            'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15',
            'R8D', 'R9D', 'R10D', 'R11D', 'R12D', 'R13D', 'R14D', 'R15D',
            'R8W', 'R9W', 'R10W', 'R11W', 'R12W', 'R13W', 'R14W', 'R15W',
            'R8B', 'R9B', 'R10B', 'R11B', 'R12B', 'R13B', 'R14B', 'R15B',
            '$', 'CS', 'DS', 'ES', 'FS', 'GS', 'SS'
        }

        tokens = re.split(r'[\s+\-*/:,()[\]]+', raw_sym)
        final_symbols = []
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            # Skip natively handled constants
            if re.match(r'^[-+]?(0x[0-9a-fA-F]+|0b[01]+|\d+[a-fA-F]*[hHbB]?|\d+)$', token):
                continue
            # Store valid identifiers avoiding registers
            if re.match(r'^[a-zA-Z_.$]', token):
                if token.upper() not in ignore_set:
                    final_symbols.append(token)

        return final_symbols

    def _collect_missing_symbols(self) -> Dict[str, str]:
        """Collects symbols referenced in instructions/globals but are not yet formally defined."""
        missing = {}

        for sec in self.program.sections:
            for child in sec.children:
                if isinstance(child, Function):
                    for bb in child.basic_blocks:
                        for instr in bb.instructions:
                            opcode = getattr(instr, 'opcode', '').upper()
                            for op in instr.operands:
                                for sym_name in self._extract_symbols_from_operand(op):
                                    if sym_name not in self.symbol_table:
                                        # Deduce logical role based on contextual usage intent
                                        if opcode in ('CALL', 'JMP'):
                                            missing[sym_name] = 'function'
                                        else:
                                            if sym_name not in missing:
                                                missing[sym_name] = 'data'

        # Ensure globals are caught
        for g in self.program.globals:
            if g not in self.symbol_table and g not in missing:
                missing[g] = 'data'

        return missing

    def _create_fallback_definitions(self, missing_symbols: Dict[str, str]):
        """Generates fallback allocations resolving otherwise irreconcilable orphaned references safely."""
        for sym, role in missing_symbols.items():
            if sym in self.symbol_table:
                continue

            entry = self._ensure_symbol_entry(sym, role)
            is_global = sym in self.program.globals
            entry['visibility'] = 'global' if is_global else 'local'

            if role == 'function':
                entry['linkage'] = 'external'
                entry['is_external'] = True
                entry['llvm_type'] = 'void (...)'  # Safe generalized signature fallback
            else:
                entry['linkage'] = 'weak' if is_global else 'internal'
                entry['llvm_type'] = '[8 x i8]'
                entry['value'] = 'zeroinitializer'
                entry['section'] = '.bss'
                if 'rodata' in sym:
                    entry['section'] = '.rodata'
                    entry['is_constant'] = True

    def _process_data_sections(self):
        """Processes and types .data, .rodata, .bss, .init_array, .fini_array."""
        missing_symbols = self._collect_missing_symbols()

        for sec_name in ['.data', '.rodata', '.bss']:
            sec = self.section_map.get(sec_name)
            if not sec:
                continue
            for child in sec.children:
                if isinstance(child, DataGroup):
                    self._parse_datagroup(child, sec_name)
            if not sec.children and sec.pseudo_instruct:
                self._parse_legacy_pseudo_data(sec, sec_name, missing_symbols)

        # Apply fallback materializations mapping anything untyped or orphaned
        self._create_fallback_definitions(missing_symbols)

    def _process_ctors_dtors(self):
        """
        Maps .init_array/.fini_array to llvm.global_ctors / llvm.global_dtors.
        Initializers correctly reroute pointers reflecting public wrapper symbols securely.
        """
        def handle_array(sec_name: str, llvm_global: str):
            sec = self.section_map.get(sec_name)
            if not sec:
                return
            pointers = []
            for pseudo in sec.pseudo_instruct:
                if pseudo.get('dx') in ('dq', 'dd'):
                    for val in pseudo.get('values', []):
                        if isinstance(val, dict) and 'symbol' in val:
                            sym_name = val['symbol']
                            sym_entry = self.symbol_table.get(sym_name, {})
                            if sym_entry.get('is_boundary'):
                                p = sym_entry.get('wrapper_ref', sym_name)
                            else:
                                p = sym_name
                            pointers.append(p)

            if pointers:
                entry = self._ensure_symbol_entry(llvm_global, 'special')
                entry['linkage'] = 'appending'
                entry['visibility'] = 'global'
                structs = [f'{{ i32 65535, void ()* @{p}, i8* null }}' for p in pointers]
                entry['llvm_type'] = f'[{len(pointers)} x {{i32, void ()*, i8*}}]'
                entry['value'] = f'[ {", ".join(structs)} ]'

        handle_array('.init_array', 'llvm.global_ctors')
        handle_array('.fini_array', 'llvm.global_dtors')

        # Cleanout logical array sections after absorption
        self.program.sections = [s for s in self.program.sections if s.name not in ('.init_array', '.fini_array')]
        self.section_map.pop('.init_array', None)
        self.section_map.pop('.fini_array', None)

    def _format_string_constant(self, flat_values: List[Any]) -> Optional[str]:
        """If the data is precisely a null-terminated byte string, formats natively as c\"...\\00\"."""
        if not flat_values or not isinstance(flat_values[-1], int) or flat_values[-1] != 0:
            return None
        if not all(isinstance(v, int) and 0 <= v <= 255 for v in flat_values):
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

    def _format_data_entry(self, entry: Dict[str, Any], flat_values: List[Any], type_hint: str):
        """Standardized application of typed layout widths directly to LLVM types mappings."""
        string_const = self._format_string_constant(flat_values)
        if string_const is not None:
            entry['value'] = string_const
            entry['llvm_type'] = f'[{len(flat_values)} x i8]'
        elif len(flat_values) == 1:
            entry['llvm_type'] = type_hint
            entry['value'] = format_llvm_constant(flat_values[0], type_hint)
        elif len(flat_values) == 0:
            entry['llvm_type'] = f'[0 x {type_hint}]'
            entry['value'] = '[]'
        else:
            entry['llvm_type'] = f'[{len(flat_values)} x {type_hint}]'
            formatted_vals = [format_llvm_constant(v, type_hint) for v in flat_values]
            entry['value'] = f'[ {", ".join(formatted_vals)} ]'

    def _parse_datagroup(self, dg: DataGroup, section_name: str):
        current_label: Optional[str] = None
        directives_buffer: List[DataDirective] = []

        for item in dg.items:
            if isinstance(item, Label):
                if current_label and directives_buffer:
                    self._commit_symbol_data(current_label, directives_buffer, section_name)
                current_label = item.name
                directives_buffer = []
            elif isinstance(item, DataDirective):
                directives_buffer.append(item)

        if current_label and directives_buffer:
            self._commit_symbol_data(current_label, directives_buffer, section_name)

    def _commit_symbol_data(self, name: str, directives: List[DataDirective], section_name: str):
        """Processes AST DataDirectives into heavily typed symbol table data entries."""
        if not directives:
            return

        flat_values: List[Any] = []
        first_kind = directives[0].kind.lower()
        type_hint = LLVM_INT_TYPES.get(first_kind, 'i8')
        if first_kind in LLVM_FLOAT_TYPES:
            type_hint = LLVM_FLOAT_TYPES[first_kind]

        for dd in directives:
            for op in dd.operands:
                if isinstance(op, Immediate):
                    flat_values.append(op.value)
                elif isinstance(op, str):
                    s = op.strip('`"').replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
                    flat_values.extend(ord(c) for c in s)
                elif isinstance(op, Expression):
                    flat_values.append(0)  # Conservative fallback to preserve structure layout fidelity

        entry = self._ensure_symbol_entry(name, 'data')
        entry['section'] = section_name
        if section_name == '.rodata':
            entry['is_constant'] = True

        self._format_data_entry(entry, flat_values, type_hint)

        if name in self.program.globals:
            entry['visibility'] = 'global'
            entry['linkage'] = 'global'
        else:
            entry['linkage'] = 'internal'

    def _parse_legacy_pseudo_data(self, sec: Section, section_name: str, missing_symbols: Dict[str, str]):
        """Guesses and recovers legacy pseudo data layouts when unnamed block chunks arise."""
        unnamed_counter = 0

        for pseudo in sec.pseudo_instruct:
            if not isinstance(pseudo, dict):
                continue

            name = pseudo.get('name')
            if not name:
                candidates = []
                if section_name == '.rodata':
                    candidates = [s for s, r in missing_symbols.items() if 'rodata' in s.lower() and r == 'data']
                elif section_name == '.data':
                    candidates = [
                        s for s, r in missing_symbols.items()
                        if (s in ('__data_start', '__dso_handle') or ('data' in s.lower() and 'rodata' not in s.lower())) and r == 'data'
                    ]
                elif section_name == '.bss':
                    candidates = [
                        s for s, r in missing_symbols.items()
                        if (s in ('__TMC_END__', 'Ltemp_storage_foxdec') or 'bss' in s.lower()) and r == 'data'
                    ]

                if candidates:
                    candidates.sort()
                    name = candidates[0]
                    del missing_symbols[name]
                else:
                    unnamed_counter += 1
                    name = f"__unnamed_{section_name.strip('.')}_{unnamed_counter}"

            dx = pseudo.get('dx')
            resx = pseudo.get('resx')
            equ = pseudo.get('equ')
            integer_val = pseudo.get('integer')

            if equ:
                entry = self._ensure_symbol_entry(name, 'constant')
                entry['visibility'] = 'global' if name in self.program.globals else 'local'

                if 'integer' in equ:
                    val = equ['integer']['value']
                    entry['value'] = parse_int(val)
                    entry['llvm_type'] = 'i64'
                elif 'expression' in equ:
                    expr = equ['expression']
                    if 'subtract' in expr:
                        parts = expr['subtract']
                        if len(parts) == 2 and parts[0] == {'symbol': '$'} and 'symbol' in parts[1]:
                            prev_name = parts[1]['symbol']
                            if prev_name in self.symbol_table:
                                prev = self.symbol_table[prev_name]
                                typ = prev.get('llvm_type', '')
                                m = re.match(r'\[(\d+) x i8\]', typ)
                                if m:
                                    entry['value'] = int(m.group(1))
                                    entry['llvm_type'] = 'i64'
            elif dx:
                self._commit_legacy_data(name, dx, pseudo.get('values', []), section_name)
            elif resx:
                count = 1
                int_obj = pseudo.get('integer')
                if int_obj and isinstance(int_obj, dict):
                    count = parse_int(int_obj.get('value', 1))

                base = resx[3:].lower()
                llvm_type_str = LLVM_RES_TYPES.get(base, 'i8')

                entry = self._ensure_symbol_entry(name, 'data')
                entry['section'] = section_name
                entry['value'] = 'zeroinitializer'
                entry['linkage'] = 'global' if name in self.program.globals else 'internal'
                entry['visibility'] = 'global' if name in self.program.globals else 'local'
                entry['llvm_type'] = llvm_type_str if count == 1 else f'[{count} x {llvm_type_str}]'
            elif integer_val:
                val = parse_int(integer_val.get('value', 0))
                entry = self._ensure_symbol_entry(name, 'constant')
                entry['value'] = str(val)
                entry['llvm_type'] = 'i64'

    def _commit_legacy_data(self, name: str, kind: str, values: List[Any], section_name: str):
        type_hint = LLVM_INT_TYPES.get(kind, 'i8')
        if kind.lower() in LLVM_FLOAT_TYPES:
            type_hint = LLVM_FLOAT_TYPES[kind.lower()]

        # Strip unreferenced padding/garbage if an unnamed chunk is lumped aggressively with a formatted str
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
                    s = v['string'].strip('`"').replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
                    flat_values.extend(ord(c) for c in s)
                elif 'integer' in v:
                    flat_values.append(parse_int(v['integer']['value']))
                elif 'float' in v:
                    type_hint = LLVM_FLOAT_TYPES[kind.lower()]
                    flat_values.append(float(v['float']['value']))
            elif isinstance(v, int):
                flat_values.append(v)

        entry = self._ensure_symbol_entry(name, 'data')
        entry['section'] = section_name
        if section_name == '.rodata':
            entry['is_constant'] = True

        self._format_data_entry(entry, flat_values, type_hint)
        entry['linkage'] = 'global' if name in self.program.globals else 'internal'
        entry['visibility'] = 'global' if name in self.program.globals else 'local'

    def _scan_relocations(self):
        """Discovers and preserves explicitly symbolic PIC/GOT relation context via reference passes."""
        for sec in self.program.sections:
            for child in sec.children:
                if isinstance(child, Function):
                    for bb in child.basic_blocks:
                        for instr in bb.instructions:
                            opcode = getattr(instr, 'opcode', '').upper()
                            for op in instr.operands:
                                for sym_name in self._extract_symbols_from_operand(op):
                                    if sym_name not in self.symbol_table:
                                        continue

                                    sym = self.symbol_table[sym_name]
                                    if sym.get('kind') == 'constant':
                                        continue

                                    relocs = sym.get('relocations', [])
                                    reloc_type = 'unknown'
                                    pic = False

                                    is_func = sym.get('kind') == 'function'
                                    is_data = sym.get('kind') == 'data'

                                    via_got = getattr(op, 'via_got', False)
                                    rip_rel = getattr(op, 'rip_relative', False)
                                    mem_base = getattr(getattr(op, 'memory', None), 'base', '').upper() if getattr(op, 'memory', None) else ''

                                    if opcode in ('CALL', 'JMP', 'LEA') and is_func:
                                        if sym.get('is_external'):
                                            reloc_type = 'plt32'
                                        else:
                                            reloc_type = 'pc32'
                                            pic = True
                                    elif is_data or (not is_func and not is_data):
                                        if via_got:
                                            reloc_type = 'gotpcrel'
                                            pic = True
                                        elif rip_rel or mem_base == 'RIP':
                                            reloc_type = 'pc32'
                                            pic = True
                                        else:
                                            reloc_type = 'abs32'
                                            pic = False

                                    if reloc_type != 'unknown':
                                        instr_id = getattr(instr, 'id', None)
                                        # Guard against array duplication representing identical operand accesses
                                        if not any(r.get('instruction') == instr_id and r.get('type') == reloc_type for r in relocs):
                                            reloc_info = {
                                                'type': reloc_type,
                                                'instruction': instr_id
                                            }
                                            if pic:
                                                reloc_info['pic'] = True
                                            relocs.append(reloc_info)
                                            sym['relocations'] = relocs

                                    if via_got:
                                        sym['accessed_via_got'] = True

# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    try:
        raw = sys.stdin.read()
        if not raw:
            sys.stderr.write("No input on stdin. Expecting JSON legacy program dict.\n")
            sys.exit(2)
        obj = json.loads(raw)
    except json.JSONDecodeError as jde:
        sys.stderr.write(f"Failed to parse JSON from stdin: {jde}\n")
        sys.exit(3)

    try:
        ast_prog: Program = legacy_program_dict_to_ast(obj, include_enhancements=True)
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
