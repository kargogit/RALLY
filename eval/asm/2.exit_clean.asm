; nasm -felf64 zero.asm && ld -o zero zero.o
global _start
section .text
_start:
    xor edi,edi     ; exit code 0
    mov eax,60      ; syscall: exit
    syscall
