section .text
global _start

_start:
    mov eax, 5          ; Load value with odd parity (binary: 101)
    test eax, eax       ; Set flags based on EAX
    jpe skip_add        ; Jump if parity even (PF=1)
    add eax, 1          ; Add 1 if parity was odd
skip_add:
    mov edi, eax        ; Exit status = result
    mov eax, 60         ; sys_exit
    syscall
