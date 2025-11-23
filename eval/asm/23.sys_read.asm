section .bss
    buffer resb 16

section .text
global _start
_start:
    mov rax, 0          ; sys_read
    mov rdi, 0          ; stdin
    lea rsi, [buffer]   ; buffer to read into
    mov rdx, 16         ; max bytes to read
    syscall

    ; rax now contains number of bytes read. Exit with that value.
    mov rdi, rax
    mov rax, 60
    syscall
