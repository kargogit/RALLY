section .text
global _start

_start:
    mov eax, 0          ; initialize counter

block1:
    nop
    nop
    inc eax             ; increment counter
    jmp block2

block2:
    nop
    nop
    mov edi, eax        ; exit with counter value
    mov eax, 60         ; sys_exit
    syscall
