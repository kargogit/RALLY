section .text
global _start

_start:
    mov eax, 5          ; first constant
    add eax, 3          ; add second constant
    mov edi, eax        ; move sum to exit status
    mov eax, 60         ; sys_exit
    syscall
