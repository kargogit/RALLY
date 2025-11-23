section .text
global _start

_start:
    mov byte [counter], 42      ; Modify the global variable
    movzx edi, byte [counter]   ; Load value for exit status
    mov eax, 60                 ; sys_exit
    syscall

section .bss
    counter resb 1              ; Mutable global variable
