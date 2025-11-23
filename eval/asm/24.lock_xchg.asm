section .data
    shared_var dd 0

section .text
global _start
_start:
    mov eax, 42
    ; Atomically swap EAX with the value at shared_var.
    ; The old value of shared_var (0) will be loaded into EAX.
    lock xchg [shared_var], eax

    ; Now EAX holds the old value (0). Exit with it.
    mov edi, eax
    mov eax, 60
    syscall
