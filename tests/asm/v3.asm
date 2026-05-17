section .data
    status_msg db "Processing...", 10
    msg_len equ $ - status_msg

    numbers db 10, 20, 30
    count   equ 3

    signed_byte db -5
    float_val   dd 2.5

section .bss
    shared_mem resd 1

section .text
global _start

_start:
    ; --- 1. Syscall: Write (Test 20) ---
    mov rax, 1
    mov rdi, 1
    mov rsi, status_msg
    mov rdx, msg_len
    syscall

    ; --- 2. Flag Logic & Carry (Test 28) ---
    mov rbx, -1
    add rbx, 1
    setc bl
    movzx eax, bl

    ; --- 3. Loop & Parity (Tests 11, 14) ---
    lea rsi, [numbers]
    mov rcx, count         ; loop counter (RCX must survive parity test)

.sum_loop:
    add al, [rsi]          ; accumulate byte in AL

    ; Parity check with no clobber of RCX
    test al, al
    setpe dl               ; use DL (not CL) so RCX is preserved
    lea r8, [.even_parity]
    lea r9, [.next_iter]
    test dl, dl
    cmove r8, r9           ; if DL == 0 (not parity even) -> move .next_iter into r8
    jmp r8                 ; indirect jump

.even_parity:
    add al, 1
    jmp .next_iter

.next_iter:
    inc rsi
    loop .sum_loop

    ; --- 4. Bitwise & Shifts (Tests 7, 10) ---
    shl eax, 2
    movsx ecx, byte [signed_byte]
    imul ecx, 2
    lea edx, [eax + ecx]

    ; --- 5. Floating Point (Test 27) ---
    movss xmm0, [float_val]
    addss xmm0, xmm0
    cvttss2si eax, xmm0
    add edx, eax

    ; --- 6. Stack & Function Call (Tests 16, 18) ---
    mov rdi, rdx           ; pass via rdi (no push/pop, preserves alignment)
    call double_val

    ; --- 7. Atomic Operations & BSS (Tests 13, 24, 26) ---
    mov edi, 100
    xchg dword [shared_mem], edi   ; xchg is atomic; no need for LOCK prefix
    lock inc dword [shared_mem]

    ; --- 8. Division & Modulo (Test 21) ---
    mov ecx, 10
    cdq
    idiv ecx

    ; --- 9. Comparisons & Overflow (Tests 5, 6, 19, 22) ---
    cmp edx, 4
    setne dl

    mov eax, 0x7FFFFFFF
    add eax, 1
    jo .overflow_ok

    mov eax, 99
    lea r8, [do_exit]
    jmp r8

.overflow_ok:
    mov eax, 1
    add eax, 44

    cmp eax, -1
    jl  error_exit

    add eax, ebx

    lea r8, [do_exit]
    jmp r8

error_exit:
    mov eax, 98
    lea r8, [do_exit]
    jmp r8

do_exit:
    mov rdi, rax
    mov rax, 60
    syscall

; --- Subroutine ---
double_val:
    mov rax, rdi
    add rax, rax
    ret
