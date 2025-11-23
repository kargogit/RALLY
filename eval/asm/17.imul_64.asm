section .text
global _start
_start:
    mov rax, -10        ; -10
    mov rdx, 5          ; 5
    imul rdx            ; rdx:rax = -50. We only care about rax for the exit code.

    mov rdi, rax        ; Exit with -50 (wrapped to 32-bit for exit code)
    mov eax, 60
    syscall
