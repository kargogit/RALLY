section .text
global _start

_start:
    lea     rsi, [numbers]      ; Load array address (RIP-relative)
    mov     ecx, 6              ; Element count
    xor     eax, eax            ; Initialize sum = 0

.sum_loop:
    add     al, [rsi]           ; Accumulate current byte
    inc     rsi                 ; Next element
    loop    .sum_loop           ; Decrement ECX and loop

    movzx   edi, al             ; Zero-extend sum to EDI (exit status)
    mov     eax, 60             ; sys_exit
    syscall

section .data
    numbers db 3, 1, 4, 1, 5, 9  ; Array of digits from π (first 6 decimals)
