extern _ITM_deregisterTMCloneTable
extern _ITM_registerTMCloneTable
extern __cxa_finalize
extern __gmon_start__
extern __libc_start_main
extern getcwd
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
; Entry 1060; block 0; address 1060
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
  LEA RDI, [main]    ; 0x1149 --> main
  CALL __libc_start_main wrt ..plt

; Entry 1060; block 1; address 1085
L1060_1:
  HLT




; ------------------------------
; Function: deregister_tm_clones
; ------------------------------
; Entry 1090; block 0; address 1090
deregister_tm_clones:
  LEA RDI, [__TMC_END__]    ; 0x4010 --> __TMC_END__
  LEA RAX, [__TMC_END__]    ; 0x4010 --> __TMC_END__
  CMP RAX, RDI
  JZ L1090_2    ; 0x10b8 --> L1090_2

; Entry 1090; block 1; address 10a3
L1090_1:
  MOV qword [Ltemp_storage_foxdec], RBX ; inserted
  LEA RBX, [_ITM_deregisterTMCloneTable wrt ..plt]
  MOV RAX, RBX
  MOV RBX, qword [Ltemp_storage_foxdec] ; inserted
  TEST RAX, RAX
  JZ L1090_2    ; 0x10b8 --> L1090_2

; Entry 1090; block 3; address 10af
L1090_3:
  ; Resolved indirection: RAX --> _ITM_deregisterTMCloneTable
  JMP _ITM_deregisterTMCloneTable wrt ..plt

; Entry 1090; block 2; address 10b8
L1090_2:
  RET




; -------------------------------
; Function: __do_global_dtors_aux
; -------------------------------
; Entry 1100; block 0; address 1100
__do_global_dtors_aux:
  CMP byte [__TMC_END__], 0x0    ; 0x4010 --> __TMC_END__
  JNZ L1100_2    ; 0x1138 --> L1100_2

; Entry 1100; block 1; address 110d
L1100_1:
  PUSH RBP
  MOV qword [Ltemp_storage_foxdec], RAX ; inserted
  LEA RAX, [__cxa_finalize wrt ..plt]
  CMP RAX, 0x0
  MOV RAX, qword [Ltemp_storage_foxdec] ; inserted
  MOV RBP, RSP
  JZ L1100_4    ; 0x1127 --> L1100_4

; Entry 1100; block 3; address 111b
L1100_3:
  MOV RDI, qword [__dso_handle]    ; 0x4008 --> __dso_handle
  CALL __cxa_finalize wrt ..plt

; Entry 1100; block 4; address 1127
L1100_4:
  CALL deregister_tm_clones    ; 0x1090 --> deregister_tm_clones

; Entry 1100; block 5; address 112c
L1100_5:
  MOV byte [__TMC_END__], 0x1    ; 0x4010 --> __TMC_END__
  POP RBP
  RET

; Entry 1100; block 2; address 1138
L1100_2:
  RET




; ---------------------
; Function: frame_dummy
; ---------------------
; Entry 1140; block 1; address 10e4
L1140_1:
  MOV qword [Ltemp_storage_foxdec], RBX ; inserted
  LEA RBX, [_ITM_registerTMCloneTable wrt ..plt]
  MOV RAX, RBX
  MOV RBX, qword [Ltemp_storage_foxdec] ; inserted
  TEST RAX, RAX
  JZ L1140_2    ; 0x10f8 --> L1140_2

; Entry 1140; block 3; address 10f0
L1140_3:
  ; Resolved indirection: RAX --> _ITM_registerTMCloneTable
  JMP _ITM_registerTMCloneTable wrt ..plt

; Entry 1140; block 2; address 10f8
L1140_2:
  RET

; Entry 1140; block 0; address 1140
frame_dummy:
  LEA RDI, [__TMC_END__]    ; 0x4010 --> __TMC_END__
  LEA RSI, [__TMC_END__]    ; 0x4010 --> __TMC_END__
  SUB RSI, RDI
  MOV RAX, RSI
  SHR RSI, 0x3f
  SAR RAX, 0x3
  ADD RSI, RAX
  SAR RSI, 0x1
  JZ L1140_2    ; 0x10f8 --> L1140_2
  JMP L1140_1 ; jump is inserted




; --------------
; Function: main
; --------------
; Entry 1149; block 0; address 1149
main:
  PUSH RBP
  MOV RBP, RSP
  SUB RSP, 0x400
  LEA RAX, [RBP - 0x400]
  MOV ESI, 0x400
  MOV RDI, RAX
  CALL getcwd wrt ..plt

; Entry 1149; block 1; address 1168
L1149_1:
  TEST RAX, RAX
  JZ L1149_3    ; 0x117c --> L1149_3

; Entry 1149; block 2; address 116d
L1149_2:
  LEA RAX, [RBP - 0x400]
  MOV RDI, RAX
  CALL puts wrt ..plt

; Entry 1149; block 3; address 117c
L1149_3:
  MOV EAX, 0x0
  LEAVE
  RET




; ---------------
; Function: _fini
; ---------------
; Entry 1184; block 0; address 1184
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
__GNU_EH_FRAME_HDR:

section .init_array align=8 ; @3db0
L_.init_array_0x3db0:
L_reloc_0x3db0_0x1140:
dq frame_dummy    ; 0x1140 --> frame_dummy
L_.init_array_END:

section .fini_array align=8 ; @3db8
L_.fini_array_0x3db8:
L_reloc_0x3db8_0x1100:
dq __do_global_dtors_aux    ; 0x1100 --> __do_global_dtors_aux
L_.fini_array_END:

section .got align=8 ; @3fb0
L_.got_0x3fb0:
db 0c0h
db `=`, 0; @ 3fb1
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
dq getcwd    ; 
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