section .text
global _start

_start:
    mov eax, -8         ; Load negative number
    mov ebx, eax        ; Copy for SAR
    shr eax, 1          ; Logical shift right
    sar ebx, 1          ; Arithmetic shift right

    ; Exit with EAX (SHR result) as low byte, EBX (SAR result) as high byte
    shl ebx, 8
    or  eax, ebx
    mov edi, eax
    mov eax, 60
    syscall
