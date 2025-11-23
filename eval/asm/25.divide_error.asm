section .text
global _start

_start:
    xor eax, eax
    div eax          ; Divide by zero - should trigger exception

    ; If we get here, exit with error
    mov edi, 99
    mov eax, 60
    syscall
