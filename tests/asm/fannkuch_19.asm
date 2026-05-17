extern _ITM_deregisterTMCloneTable
extern _ITM_registerTMCloneTable
extern __cxa_finalize
extern __gmon_start__
extern __libc_start_main
extern puts


global _IO_stdin_used
global __TMC_END__
global __data_start
global __dso_handle
global _end
global _fini
global _init
global _start
global main


default rel


; ---------------
; Function: _init
; ---------------
; Entry 1000; block 0; address 1000
_init:
  SUB RSP, 0x8
  MOV qword [Ltemp_storage_foxdec], RBX ; inserted
  LEA RBX, [__gmon_start__ wrt ..plt]
  MOV RAX, RBX
  MOV RBX, qword [Ltemp_storage_foxdec] ; inserted
  TEST RAX, RAX
  JZ L1000_2    ; 0x1016 --> L1000_2

; Entry 1000; block 1; address 1014
L1000_1:
  ; Resolved indirection: RAX --> __gmon_start__
  CALL __gmon_start__ wrt ..plt

; Entry 1000; block 2; address 1016
L1000_2:
  ADD RSP, 0x8
  RET




; ----------------
; Function: _start
; ----------------
; Entry 1050; block 0; address 1050
_start:
  XOR EBP, EBP
  MOV R9, RDX
  POP RSI
  MOV RDX, RSP
  AND RSP, 0xfffffffffffffff0
  PUSH RAX
  PUSH RSP
  XOR R8D, R8D
  XOR ECX, ECX
  LEA RDI, [main]    ; 0x1139 --> main
  CALL __libc_start_main wrt ..plt

; Entry 1050; block 1; address 1075
L1050_1:
  HLT




; ------------------------------
; Function: deregister_tm_clones
; ------------------------------
; Entry 1080; block 0; address 1080
deregister_tm_clones:
  LEA RDI, [__TMC_END__]    ; 0x4010 --> __TMC_END__
  LEA RAX, [__TMC_END__]    ; 0x4010 --> __TMC_END__
  CMP RAX, RDI
  JZ L1080_2    ; 0x10a8 --> L1080_2

; Entry 1080; block 1; address 1093
L1080_1:
  MOV qword [Ltemp_storage_foxdec], RBX ; inserted
  LEA RBX, [_ITM_deregisterTMCloneTable wrt ..plt]
  MOV RAX, RBX
  MOV RBX, qword [Ltemp_storage_foxdec] ; inserted
  TEST RAX, RAX
  JZ L1080_2    ; 0x10a8 --> L1080_2

; Entry 1080; block 3; address 109f
L1080_3:
  ; Resolved indirection: RAX --> _ITM_deregisterTMCloneTable
  JMP _ITM_deregisterTMCloneTable wrt ..plt

; Entry 1080; block 2; address 10a8
L1080_2:
  RET




; -------------------------------
; Function: __do_global_dtors_aux
; -------------------------------
; Entry 10f0; block 0; address 10f0
__do_global_dtors_aux:
  CMP byte [__TMC_END__], 0x0    ; 0x4010 --> __TMC_END__
  JNZ L10f0_2    ; 0x1128 --> L10f0_2

; Entry 10f0; block 1; address 10fd
L10f0_1:
  PUSH RBP
  MOV qword [Ltemp_storage_foxdec], RAX ; inserted
  LEA RAX, [__cxa_finalize wrt ..plt]
  CMP RAX, 0x0
  MOV RAX, qword [Ltemp_storage_foxdec] ; inserted
  MOV RBP, RSP
  JZ L10f0_4    ; 0x1117 --> L10f0_4

; Entry 10f0; block 3; address 110b
L10f0_3:
  MOV RDI, qword [__dso_handle]    ; 0x4008 --> __dso_handle
  CALL __cxa_finalize wrt ..plt

; Entry 10f0; block 4; address 1117
L10f0_4:
  CALL deregister_tm_clones    ; 0x1080 --> deregister_tm_clones

; Entry 10f0; block 5; address 111c
L10f0_5:
  MOV byte [__TMC_END__], 0x1    ; 0x4010 --> __TMC_END__
  POP RBP
  RET

; Entry 10f0; block 2; address 1128
L10f0_2:
  RET




; ---------------------
; Function: frame_dummy
; ---------------------
; Entry 1130; block 1; address 10d4
L1130_1:
  MOV qword [Ltemp_storage_foxdec], RBX ; inserted
  LEA RBX, [_ITM_registerTMCloneTable wrt ..plt]
  MOV RAX, RBX
  MOV RBX, qword [Ltemp_storage_foxdec] ; inserted
  TEST RAX, RAX
  JZ L1130_2    ; 0x10e8 --> L1130_2

; Entry 1130; block 3; address 10e0
L1130_3:
  ; Resolved indirection: RAX --> _ITM_registerTMCloneTable
  JMP _ITM_registerTMCloneTable wrt ..plt

; Entry 1130; block 2; address 10e8
L1130_2:
  RET

; Entry 1130; block 0; address 1130
frame_dummy:
  LEA RDI, [__TMC_END__]    ; 0x4010 --> __TMC_END__
  LEA RSI, [__TMC_END__]    ; 0x4010 --> __TMC_END__
  SUB RSI, RDI
  MOV RAX, RSI
  SHR RSI, 0x3f
  SAR RAX, 0x3
  ADD RSI, RAX
  SAR RSI, 0x1
  JZ L1130_2    ; 0x10e8 --> L1130_2
  JMP L1130_1 ; jump is inserted




; --------------
; Function: main
; --------------
; Entry 1139; block 0; address 1139
main:
  PUSH RBP
  MOV RBP, RSP
  SUB RSP, 0x10
  MOV dword [RBP - 0x4], 0x1
  CMP dword [RBP - 0x4], 0x0
  SETNZ AL
  MOVZX EAX, AL
  TEST RAX, RAX
  JZ L1139_2    ; 0x1166 --> L1139_2

; Entry 1139; block 1; address 1157
L1139_1:
  LEA RAX, [L_.rodata_0x2004]    ; 0x2004 --> L_.rodata_0x2004
  MOV RDI, RAX
  CALL puts wrt ..plt

; Entry 1139; block 2; address 1166
L1139_2:
  MOV EAX, 0x0
  LEAVE
  RET




; ---------------
; Function: _fini
; ---------------
; Entry 1170; block 0; address 1170
_fini:
  SUB RSP, 0x8
  ADD RSP, 0x8
  RET




section .rodata align=4 ; @2000
L_.rodata_0x2000:
db 01h
db 00h
db 02h
db 00h
L_.rodata_0x2004:
db `Likely branch taken`, 0; @ 2004
__GNU_EH_FRAME_HDR:

section .init_array align=8 ; @3db8
L_.init_array_0x3db8:
L_reloc_0x3db8_0x1130:
dq frame_dummy    ; 0x1130 --> frame_dummy
L_.init_array_END:

section .fini_array align=8 ; @3dc0
L_.fini_array_0x3dc0:
L_reloc_0x3dc0_0x10f0:
dq __do_global_dtors_aux    ; 0x10f0 --> __do_global_dtors_aux
L_.fini_array_END:

section .got align=8 ; @3fb8
L_.got_0x3fb8:
db 0c8h
db `=`, 0; @ 3fb9
db 00h
db 00h
db 00h
db 00h
db 00h
db 00h
db 00h
db 00h
db 00h
db 00h
db 00h
db 00h
db 00h
db 00h
db 00h
db 00h
db 00h
db 00h
db 00h
db 00h
db 00h
dq puts    ; 
dq __libc_start_main    ; 
dq _ITM_deregisterTMCloneTable    ; 
dq __gmon_start__    ; 
dq _ITM_registerTMCloneTable    ; 
dq __cxa_finalize    ; 
L_.got_END:




section .data align=8 ; @4000
__data_start:
db 00h
db 00h
db 00h
db 00h
db 00h
db 00h
db 00h
db 00h
__dso_handle:
dq __dso_handle    ; 0x4008 --> __dso_handle
L_.data_END:




section .bss align=1 ; @4010
__TMC_END__:
resb 8
L_.bss_END:









section .bss
Ltemp_storage_foxdec:
resb 8