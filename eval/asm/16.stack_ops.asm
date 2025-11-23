section .text
global _start

_start:
    mov rax, 0x1111111111111111
    mov rbx, 0x2222222222222222
    push rax            ; Stack: [0x11...]
    push rbx            ; Stack: [0x22..., 0x11...]

    pop rdi             ; Pops 0x22... into RDI. Exit status will be 0x22.

    mov eax, 60
    syscall
