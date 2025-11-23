section .text
global _start
_start:
    mov rax, -1         ; 0xFFFFFFFFFFFFFFFF
    add rax, 1          ; Result is 0, Carry Flag (CF) is set to 1

    setc al             ; Set AL to 1 if CF=1, otherwise 0. AL will be 1.
    movzx edi, al       ; Exit with 1
    mov eax, 60
    syscall
