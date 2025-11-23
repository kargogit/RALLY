section .text
global _start

_start:
    movss xmm0, [val1]
    addss xmm0, [val2]
    mulss xmm0, [val3]
    cvttss2si edi, xmm0

    mov eax, 60
    syscall

section .data
    val1 dd 2.5
    val2 dd 1.5
    val3 dd 2.0
