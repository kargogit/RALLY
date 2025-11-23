section .text
global _start

_start:
    lea     rsi, [message]      ; Load string address (RIP-relative)
    xor     ecx, ecx            ; Initialize length counter = 0

.length_loop:
    cmp     byte [rsi + rcx], 0 ; Check for null terminator
    je      .done               ; End of string reached
    inc     ecx                 ; Increment length
    jmp     .length_loop        ; Continue loop

.done:
    mov     edi, ecx            ; Exit status = string length
    mov     eax, 60             ; sys_exit
    syscall

section .data
    message db "Test", 0        ; Null-terminated string
