section .text
global _start

_start:
    movzx eax, byte [yolo]    ; Zero-extend byte to dword
    and eax, 0x0F              ; Mask with AND (keep lower 4 bits)

    mov edi, eax               ; Exit status = result
    mov eax, 60                ; sys_exit
    syscall

section .data
    yolo db 0xAB              ; Example byte value
