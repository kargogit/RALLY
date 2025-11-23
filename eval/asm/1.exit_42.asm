section .text
global _start

_start:
    mov eax, 60      ; sys_exit system call number
    mov edi, 42      ; exit status
    syscall          ; invoke system call
