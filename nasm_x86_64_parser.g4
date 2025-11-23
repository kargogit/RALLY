parser grammar nasm_x86_64_parser;

options {
    caseInsensitive = true;
    tokenVocab = nasm_x86_64_lexer;
}

program
    : (block | line)* EOF
    ;

line
    : label? (directive | times_prefix? (pseudoinstruction | instruction))? EOL
    ;

label
    : name COLON
    ;

block
    : label EOL (EOL | non_terminator_line | terminator_line)+
    ;

non_terminator_line
    : instruction EOL
    ;

terminator_line
    : terminator_instruction EOL
    ;

terminator_instruction
    : terminator_opcode operand? (COMMA operand)*
    ;

terminator_opcode
    : CALL
    | JMP
    | RET
    | HLT
    | JA
    | JAE
    | JB
    | JBE
    | JC
    | JE
    | JG
    | JGE
    | JL
    | JLE
    | JNA
    | JNAE
    | JNB
    | JNBE
    | JNC
    | JNE
    | JNG
    | JNGE
    | JNL
    | JNLE
    | JNO
    | JNP
    | JNS
    | JNZ
    | JO
    | JP
    | JPE
    | JPO
    | JS
    | JZ
    | JCXZ
    | JECXZ
    | JRCXZ
    ;

directive
    : bits decimal_integer
    | use16
    | use32
    | default default_perfix
    | section section_params
    | absolute (integer | name)
    | (extern | required) extern_params
    | global global_params
    | common common_params
    | static name
    | cpu (decimal_integer | name)
    | float_name float_params
    | LEFT_BRACKET (warning warning_state? warning_class | map map_type name) RIGHT_BRACKET
    | org integer
    | (group name name | import_rule) name+
    | uppercase
    | export export_params
    | safeseh name
    | osabi decimal_integer
    ;

bits
    : BITS
    ;

decimal_integer
    : DECIMAL_INTEGER
    ;

use16
    : USE16
    ;

use32
    : USE32
    ;

default
    : DEFAULT
    ;

default_perfix
    : REL
    | ABS
    | BND
    | NOBND
    ;

section
    : SECTION
    | SEGMENT
    ;

section_params
    : name attribute? section_type? class? overlay? designation? allocation? execution? writing? starting_possition? follow? (
        use16
        | use32
    )? flat? (absolute_seg | alignment)? comdat? tls?
    ;

name
    : NAME
    ;

attribute
    : PRIVATE
    | PUBLIC
    | COMMON
    | STACK
    ;

section_type
    : CODE
    | TEXT
    | DATA
    | BSS
    | RDATA
    | INFO
    | MIXED
    | ZEROFILL
    | NO_DEAD_STRIP
    | LIVE_SUPPORT
    | STRIP_STATIC_SYMS
    | DEBUG
    ;

class
    : CLASS_ EQUAL_1 name
    ;

overlay
    : OVERLAY EQUAL_1 name
    ;

designation
    : PROGBITS
    | NOBITS
    | NOTE
    | PREINIT_ARRAY
    | INIT_ARRAY
    | FINI_ARRAY
    ;

allocation
    : ALLOC
    | NOALLOC
    ;

execution
    : EXEC
    | NOEXEC
    ;

writing
    : WRITE
    | NOWRITE
    ;

starting_possition
    : VSTART EQUAL_1 integer
    ;

follow
    : (FOLLOWS | VFOLLOWS) EQUAL_1 name
    ;

flat
    : FLAT
    ;

absolute_seg
    : ABSOLUTE EQUAL_1 integer
    ;

alignment
    : (ALIGN | START) EQUAL_1 integer
    | POINTER
    ;

comdat
    : COMDAT EQUAL_1 integer COLON name
    ;

tls
    : TLS
    ;

absolute
    : ABSOLUTE
    ;

integer
    : DECIMAL_INTEGER
    | OCT_INTEGER
    | HEX_INTEGER
    | BIN_INTEGER
    ;

extern
    : EXTERN
    ;

extern_params
    : name (COLON (wrt | weak))? (COMMA extern_params)*
    ;

wrt
    : name? WRT (name | DOT_DOT_PLT) (COLON integer)?
    ;

weak
    : WEAK
    ;

required
    : REQUIRED
    ;

global
    : GLOBAL
    ;

global_params
    : name (COLON global_type)? visibility? binding? expression? (COMMA global_params)*
    ;

global_type
    : FUNCTION
    | DATA
    | OBJECT
    ;

visibility
    : DEFAULT
    | INTERNAL
    | HIDDEN_
    | PROTECTED
    ;

binding
    : WEAK
    | STRONG
    ;

common
    : COMMON
    ;

common_params
    : name (integer (COLON (near | far | integer | wrt))?)+ integer?
    ;

near
    : NEAR
    ;

far
    : FAR
    ;

static
    : STATIC
    ;

cpu
    : CPU
    ;

float_name
    : FLOAT_NAME
    ;

float_params
    : DAZ
    | NODAZ
    | NEAR
    | UP
    | DOWN
    | ZERO
    | DEFAULT
    ;

warning
    : WARNING
    ;

warning_state
    : PLUS
    | MINUS
    | MULTIPLICATION
    ;

warning_class
    : warning_name
    | push
    | pop
    ;

warning_name
    : WARNING_NAME
    | NAME
    ;

push
    : PUSH
    ;

pop
    : POP
    ;

org
    : ORG
    ;

map
    : MAP
    ;

map_type
    : ALL
    | BRIEF
    | SECTIONS
    | SEGMENTS
    | SYMBOLS
    ;

group
    : GROUP
    ;

uppercase
    : UPPERCASE
    ;

import_rule
    : IMPORT
    ;

export
    : EXPORT
    ;

export_params
    : name name? (resident | nodata | parm | integer)*
    ;

resident
    : RESIDENT
    ;

nodata
    : NODATA
    ;

parm
    : PARM EQUAL_1 integer
    ;

safeseh
    : SAFESEH
    ;

osabi
    : OSABI
    ;

times_prefix
    : times (expression | integer)
    ;

times
    : TIMES
    ;

pseudoinstruction
    : name? (
        dx value (COMMA value)*
        | resx integer
        | incbin atom (COMMA atom)*
        | equ (integer | expression)
    )
    ;

dx
    : DB
    | DW
    | DD
    | DQ
    | DT
    | DO
    | DY
    | DZ
    ;

float_number
    : FLOAT_NUMBER
    ;

question
    : QUESTION
    ;

resx
    : RESB
    | RESW
    | RESD
    | RESQ
    | REST
    | RESO
    | RESY
    | RESZ
    ;

incbin
    : INCBIN
    ;

string
    : STRING
    ;

value
    : atom
    | size value
    | list
    | macro_call
    ;

atom
    : integer
    | float_number
    | string
    | name
    | question
    | expression
    ;

size
    : BYTE
    | WORD
    | DWORD
    | QWORD
    | TWORD
    | OWORD
    | YWORD
    | ZWORD
    ;

list
    : duplist
    | PERCENT parlist
    | size PERCENT? parlist
    ;

duplist
    : expression DUP size? PERCENT? parlist
    ;

parlist
    : LEFT_PARENTHESIS value (COMMA value)* RIGHT_PARENTHESIS
    ;

unaryExpression
    : unaryOperator castExpression
    | LEFT_PARENTHESIS expression RIGHT_PARENTHESIS
    ;

unaryOperator
    : PLUS
    | MINUS
    | BITWISE_NOT
    | BOOLEAN_NOT
    ;

castExpression
    : unaryExpression
    | integer
    | register
    | (register COLON)? name
    | string
    | float_number
    | DOLLAR
    | DOUBLE_DOLLAR
    ;

multiplicativeExpression
    : castExpression (
        (MULTIPLICATION | UNSIGNED_DIVISION | SIGNED_DIVISION | PERCENT | SIGNED_MODULE) castExpression
    )*
    ;

additiveExpression
    : castExpression ((PLUS | MINUS) castExpression)*
    | multiplicativeExpression ((PLUS | MINUS) multiplicativeExpression)*
    ;

shiftExpression
    : additiveExpression (
        (LEFT_SHIFT | RIGHT_SHIFT | LEFT_SHIFT_COMPLETENESS | RIGHT_SHIFT_COMPLETENESS) additiveExpression
    )*
    ;

expression
    : castExpression
    | additiveExpression
    | shiftExpression (QUESTION integer COLON expression)?
    | wrt
    ;

equ
    : EQU
    ;

lock_prefix
    : LOCK
    ;

repeat_prefix
    : REP
    | REPE
    | REPZ
    | REPNE
    | REPNZ
    ;

instruction
    : (lock_prefix | repeat_prefix)? opcode operand? (COMMA operand)*
    | macro_call
    ;

opcode
    : MOV | LEA
    | PUSH | POP                    // Stack operations
    | ADD | SUB | INC | DEC         // Arithmetic
    | MOVSS | ADDSS | MULSS         // Floating-Point Arithmetic (SSE)
    | CVTTSS2SI                     // Floating-Point to Integer Conversion (SSE)
    | CMP | TEST | SETNE | SETC     // Comparison
    | AND | OR | XOR                // Logical
    | SHL | SHR | SAR               // Bit shifts
    | CALL | JMP | RET              // Control flow (if not in terminator_instruction)
    | NOP                           // No operation
    | MOVSX | MOVZX | MOVSXD        // Sign/zero extend (essential for type conversions)
    | IMUL | MUL                    // Multiplication
    | IDIV | DIV                    // Division
    | NEG                           // Negation
    | XCHG                          // Exchange
    | LOOP | LOOPE | LOOPNE         // Loop control
    | SYSCALL                       // System calls (Linux/Unix PIC)
    | NOT                           // Bitwise NOT
    | ROL | ROR                     // Rotate
    | BT | BTS | BTR | BTC          // Bit test and set
    | LEA                           // RIP-relative addressing
    | UD2                           // Undefined instruction (for deliberate faults)
    ;

segment_register
    : ES
    | CS
    | SS
    | DS
    | FS
    | GS
    ;

operand
    : (register COLON)? (register | name)
    | (strict? size)? (
        string
        | float_number
        | integer
        | LEFT_BRACKET (segment_register COLON)? expression (COMMA expression)* RIGHT_BRACKET
    )
    | expression
    ;

register
    : RSP | ESP | SP            // Stack operations
    | RAX | EAX | AX | AL | AH  // Syscalls/accumulation
    | RDI | EDI | DI | DIL      // First syscall arg
    | RSI | ESI | SI | SIL      // Second syscall arg (RIP-relative loads)
    | RDX | EDX | DX | DL       // Third syscall arg
    | RBX | EBX | BX | BL
    | RCX | ECX | CX | CL
    | RBP | EBP | BP
    | R8  | R8D  | R8W  | R8B
    | R9  | R9D  | R9W  | R9B
    | R10 | R10D | R10W | R10B
    | R11 | R11D | R11W | R11B
    | R12 | R12D | R12W | R12B
    | R13 | R13D | R13W | R13B
    | R14 | R14D | R14W | R14B
    | R15 | R15D | R15W | R15B
    | XMM0 | XMM1 | XMM2 | XMM3 | XMM4 | XMM5 | XMM6 | XMM7
    | XMM8 | XMM9 | XMM10 | XMM11 | XMM12 | XMM13 | XMM14 | XMM15
    ;

strict
    : STRICT
    ;

macro_call
    : name (
        macro_param? (COMMA macro_param)*
        | LEFT_PARENTHESIS macro_param? (COMMA macro_param)* RIGHT_PARENTHESIS
    )
    ;

macro_param
    : string
    | name
    | integer
    | float_number
    ;
