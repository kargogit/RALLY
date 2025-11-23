section .text
global _start
_start:
    mov eax, 0x7FFFFFFF ; Max int32
    add eax, 1          ; Causes signed overflow (OF=1)
    jo  overflowed      ; Jump if overflow
    mov edi, 0          ; Should not reach
    jmp done
overflowed:
    mov edi, 1          ; Exit 1 if overflow detected
done:
    mov eax, 60
    syscall
