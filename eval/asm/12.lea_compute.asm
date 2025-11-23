section .text
global _start

_start:
    lea eax, [rax + rax*2 + 3]  ; example: base=rax(0), index=rax(0) → 0+0*2+3 = 3
    mov edi, eax                ; exit with computed offset
    mov eax, 60                 ; sys_exit
    syscall
