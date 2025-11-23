section .text
global _start

_start:
    mov eax, 0x0F       ; first constant
    and eax, 0x03       ; bitwise AND with second constant
    or  eax, 0x08       ; bitwise OR with third constant
    xor eax, 0x01       ; bitwise XOR with fourth constant
    mov edi, eax        ; result as exit status
    mov eax, 60         ; sys_exit
    syscall
