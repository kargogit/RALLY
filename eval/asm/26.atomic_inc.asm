section .text
global _start

_start:
    mov byte [counter], 0
    lock inc byte [counter]
    lock inc byte [counter]

    movzx edi, byte [counter]
    mov eax, 60
    syscall

section .bss
    counter resb 1
