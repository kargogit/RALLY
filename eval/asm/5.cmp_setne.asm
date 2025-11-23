section .text
global _start

_start:
    mov eax, 5          ; first value
    cmp eax, 6          ; compare with second value
    setne al            ; set AL=1 if not equal, AL=0 if equal
    movzx edi, al       ; zero-extend AL to EDI for exit status
    mov eax, 60         ; sys_exit
    syscall
