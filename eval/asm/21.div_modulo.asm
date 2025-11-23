section .text
global _start
_start:
    mov edx, 0          ; Clear dividend high bits
    mov eax, 17         ; Dividend = 17
    mov ecx, 5          ; Divisor = 5
    div ecx             ; EAX = 3 (quotient), EDX = 2 (remainder)
    mov edi, edx        ; Exit with remainder (2)
    mov eax, 60
    syscall
