section .text
global _start
_start:
    mov rax, 1          ; sys_write
    mov rdi, 1          ; stdout
    lea rsi, [msg]      ; message address
    mov rdx, 13         ; message length
    syscall

    ; Now exit cleanly
    mov rax, 60
    xor rdi, rdi
    syscall

section .data
    msg db "Hello World!", 10 ; 10 is newline
