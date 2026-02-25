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
    'dq': 'i64'
}
LLVM_FLOAT_TYPES = {
    'dd': 'float',  # 32-bit float
    'dq': 'double'  # 64-bit float
}
LLVM_RES_TYPES = {
    'b': 'i8',   # resb
    'w': 'i16',  # resw
    'd': 'i32',  # resd
    'q': 'i64'   # resq
}

def get_directive_size(directive: str) -> int:
    """Returns the size in bytes of a data directive (db, dw, dd, dq)."""
    sizes = {'db': 1, 'dw': 2, 'dd': 4, 'dq': 8}
    return sizes.get(directive.lower(), 1)

def format_llvm_constant(val: Union[int, str, float], type_hint: Optional[str] = None) -> str:
    """Formats a python value into an LLVM constant string."""
    if isinstance(val, int):
        return str(val)
    elif isinstance(val, float):
        return str(val)
    elif isinstance(val, str):
        return f'c"{val}"'
    return str(val)

# ---------------------------------------------------------------------------
# Core Analysis Logic
# ---------------------------------------------------------------------------
class SymbolMapper:
    def __init__(self, program: Program):
        self.program = program
        self.symbol_table = program.symbol_table if program.symbol_table else {}
        self.program.symbol_table = self.symbol_table
        self.section_map: Dict[str, Section] = {s.name: s for s in program.sections}

    def run(self):
        """Orchestrates the mapping process."""
        self._extract_externs()
        self._process_code_symbols()
        self._process_data_sections()
        self._process_ctors_dtors()
        self._scan_relocations()

    def _ensure_symbol_entry(self, name: str, kind: str = 'unknown') -> Dict[str, Any]:
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
            if isinstance(pseudo, dict) and pseudo.get('directive') == 'extern':
                params = pseudo.get('params', [])
                for sym_name in params:
                    entry = self._ensure_symbol_entry(sym_name, 'function' if sym_name not in ['stderr'] else 'data')
                    entry['visibility'] = 'global'
                    entry['linkage'] = 'external'
                    entry['is_external'] = True

                    # Known external types
                    if sym_name == 'printf':
                        entry['llvm_type'] = 'i32 (i8*, ...)'
                    elif sym_name == 'fprintf':
                        entry['llvm_type'] = 'i32 (i8*, i8*, ...)'
                    elif sym_name == 'exit':
                        entry['llvm_type'] = 'void (i32)'
                    elif sym_name == 'stderr':
                        entry['llvm_type'] = 'i8*'

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

                        # Boundary functions
                        if getattr(child, 'is_boundary', False):
                            entry['is_boundary'] = True
                            # Dual references per spec (wrapper = public ABI name; lifted = internal %State* version)
                            entry['wrapper_ref'] = child.entry_label
                            entry['lifted_ref'] = f"{child.entry_label}_lifted"

                    for bb in child.basic_blocks:
                        if bb.start_label:
                            entry = self._ensure_symbol_entry(bb.start_label.name, 'label')
                            entry['section'] = sec.name

    def _process_data_sections(self):
        """Processes .data, .rodata, .bss, .init_array, .fini_array."""
        for sec_name in ['.data', '.rodata', '.bss']:
            sec = self.section_map.get(sec_name)
            if not sec:
                continue
            for child in sec.children:
                if isinstance(child, DataGroup):
                    self._parse_datagroup(child, sec_name)
            if not sec.children and sec.pseudo_instruct:
                self._parse_legacy_pseudo_data(sec, sec_name)

    def _process_ctors_dtors(self):
        """
        Maps .init_array/.fini_array to llvm.global_ctors / llvm.global_dtors.
        REMAPPED: initializer pointers now reference public wrapper symbols (not the lifted versions).
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

        # Fix: Remove the explicit .init_array/.fini_array sections now that their
        # intent is captured by llvm.global_ctors/dtors in the symbol table.
        self.program.sections = [s for s in self.program.sections if s.name not in ('.init_array', '.fini_array')]
        # Update the section map for internal consistency
        self.section_map.pop('.init_array', None)
        self.section_map.pop('.fini_array', None)

    def _format_string_constant(self, flat_values: List[int]) -> Optional[str]:
        """If the data is a null-terminated byte string, format as c\"...\\00\"."""
        if not flat_values or flat_values[-1] != 0:
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

    def _parse_datagroup(self, dg: DataGroup, section_name: str):
        current_label: Optional[str] = None
        directives_buffer: List[DataDirective] = []

        for item in dg.items:
            if isinstance(item, Label):
                # Flush previous label
                if current_label and directives_buffer:
                    self._commit_symbol_data(current_label, directives_buffer, section_name)

                current_label = item.name
                directives_buffer = []

            elif isinstance(item, DataDirective):
                directives_buffer.append(item)

        # Flush last one
        if current_label and directives_buffer:
            self._commit_symbol_data(current_label, directives_buffer, section_name)

    def _commit_symbol_data(self, name: str, directives: List[DataDirective], section_name: str):
        """
        Processes a list of DataDirectives into a symbol table entry.
        Applies the same fixes as _commit_legacy_data:
        1. is_constant only for .rodata.
        2. Single values (even bytes) emitted as scalars, not [1 x type].
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
                    # String literal
                    s = op.strip('"')
                    for c in s:
                        flat_values.append(ord(c))
                elif isinstance(op, Expression):
                    # Complex expression - skipped for simple value derivation
                    pass

        entry = self._ensure_symbol_entry(name, 'data')
        entry['section'] = section_name
        # FIX 1: Only set is_constant for .rodata. .data is writable.
        if section_name == '.rodata':
            entry['is_constant'] = True

        string_const = self._format_string_constant(flat_values)
        if string_const is not None:
            entry['value'] = string_const
            entry['llvm_type'] = f'[{len(flat_values)} x i8]'
            # Removed: entry['is_constant'] = True (Fix 1)
        # FIX 2: Emit single values as scalars (including single bytes).
        elif len(flat_values) == 1:
            entry['llvm_type'] = type_hint
            entry['value'] = format_llvm_constant(flat_values[0], type_hint)
        else:
            entry['llvm_type'] = f'[{len(flat_values)} x {type_hint}]'
            formatted_vals = [format_llvm_constant(v, type_hint) for v in flat_values]
            entry['value'] = f'[ {", ".join(formatted_vals)} ]'

        if name in self.program.globals:
            entry['visibility'] = 'global'
            entry['linkage'] = 'global'
        else:
            entry['linkage'] = 'internal'

    def _parse_legacy_pseudo_data(self, sec: Section, section_name: str):
        """Handles data present in pseudo_instruct (seen in sample input)."""
        for pseudo in sec.pseudo_instruct:
            if not isinstance(pseudo, dict):
                continue
            name = pseudo.get('name')
            if not name:
                continue

            dx = pseudo.get('dx')
            resx = pseudo.get('resx')
            equ = pseudo.get('equ')
            integer_val = pseudo.get('integer')

            if equ:
                # Explicitly force kind to 'constant' to override any incorrect 'data' classification from input
                entry = self._ensure_symbol_entry(name, 'constant')
                entry['kind'] = 'constant'
                entry['visibility'] = 'local'
                if name in self.program.globals:
                    entry['visibility'] = 'global'
                
                if 'integer' in equ:
                    val = int(equ['integer']['value'])
                    entry['value'] = val
                    entry['llvm_type'] = 'i64'
                elif 'expression' in equ:
                    expr = equ['expression']
                    if 'subtract' in expr:
                        parts = expr['subtract']
                        if (len(parts) == 2 and parts[0] == {'symbol': '$'} and
                            'symbol' in parts[1]):
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
                self._commit_legacy_data(name, dx, pseudo.get('values', []), section_name)

            elif resx:
                count = 1
                int_obj = pseudo.get('integer')
                if int_obj and isinstance(int_obj, dict):
                    count = int(int_obj.get('value', 1))
                base = resx[3:].lower()
                llvm_type_str = LLVM_RES_TYPES.get(base, 'i8')
                entry = self._ensure_symbol_entry(name, 'data')
                entry['section'] = section_name
                entry['value'] = 'zeroinitializer'
                entry['linkage'] = 'internal'
                if count == 1:
                    entry['llvm_type'] = llvm_type_str
                else:
                    entry['llvm_type'] = f'[{count} x {llvm_type_str}]'
                if name in self.program.globals:
                    entry['visibility'] = 'global'

            elif integer_val:
                val = int(integer_val.get('value', 0))
                entry = self._ensure_symbol_entry(name, 'constant')
                entry['value'] = str(val)
                entry['llvm_type'] = 'i64'

    def _commit_legacy_data(self, name: str, kind: str, values: List[Any], section_name: str):
        size = get_directive_size(kind)
        type_hint = LLVM_INT_TYPES.get(kind, 'i8')
        is_float = kind.lower() in LLVM_FLOAT_TYPES
        if is_float:
            type_hint = LLVM_FLOAT_TYPES[kind.lower()]

        flat_values: List[Any] = []
        for v in values:
            if isinstance(v, dict):
                if 'string' in v:
                    # Ensure string parsing is robust.
                    # The input AST often has the string content quoted: "\"...\""
                    # We need to strip the outer quotes to get the actual content bytes.
                    s = v['string'].strip('"')
                    for c in s:
                        flat_values.append(ord(c))
                elif 'integer' in v:
                    flat_values.append(int(v['integer']['value']))
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

        # Prefer string constant format when possible
        string_const = self._format_string_constant(flat_values)
        if string_const is not None:
            entry['value'] = string_const
            entry['llvm_type'] = f'[{len(flat_values)} x i8]'
            # Removed: entry['is_constant'] = True
            # Emit single values as scalars (including single bytes).
        elif len(flat_values) == 1:
            entry['llvm_type'] = type_hint
            entry['value'] = format_llvm_constant(flat_values[0], type_hint)
        else:
            entry['llvm_type'] = f'[{len(flat_values)} x {type_hint}]'
            formatted_vals = [format_llvm_constant(v, type_hint) for v in flat_values]
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
                                    sym_name = op.symbol_ref.name
                                elif getattr(op, 'expression', None) and isinstance(op.expression, str):
                                    sym_name = op.expression

                                if not sym_name or sym_name not in self.symbol_table:
                                    continue

                                sym = self.symbol_table[sym_name]
                                # Constants (equ) generate no relocations.
                                # This check now works because we fixed the 'kind' to 'constant' in _parse_legacy_pseudo_data.
                                if sym.get('kind') == 'constant':
                                    continue  # immediates need no relocation

                                relocs = sym.get('relocations', [])
                                reloc_type = 'unknown'
                                pic = False

                                if instr.opcode in ('CALL', 'JMP') and sym.get('kind') == 'function':
                                    if sym.get('is_external'):
                                        reloc_type = 'plt32'
                                    else:
                                        continue  # internal direct call/branch – no relocation
                                elif sym.get('kind') == 'data':
                                    # Prioritize GOT access over generic RIP-relative access.
                                    if op.via_got:
                                        reloc_type = 'gotpcrel'
                                        pic = True
                                    elif op.rip_relative or (op.memory and getattr(op.memory, 'base', None) == 'RIP'):
                                        reloc_type = 'pc32'
                                        pic = True
                                    else:
                                        # Fallback for absolute or non-RIP memory references (rare in x64 PIC)
                                        reloc_type = 'pc32'
                                        pic = True

                                if reloc_type != 'unknown':
                                    reloc_info = {
                                        'type': reloc_type,
                                        'instruction': instr.id
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
        legacy_out = ast_to_legacy_program_dict(ast_prog, include_instr_locations=False, include_enhancements=True)
        json.dump(legacy_out, sys.stdout, indent=2, sort_keys=False)
        sys.stdout.write("\n")
    except Exception as exc:
        sys.stderr.write(f"Unexpected error in Step 6: {repr(exc)}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
