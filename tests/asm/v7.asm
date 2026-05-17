default rel

section .init_array
    dq init_func

section .fini_array
    dq fini_func

section .rodata
    status_msg          db "Processing...", 10, 0
    usage_fmt           db "Usage: %s [options]", 10, 0
    options_line        db "Options:", 0
    help_option_line    db "  -h, --help    Show this help message and exit", 0
    invalid_msg         db "Invalid option: %s", 10, 0
    help_short          db "-h", 0
    help_long           db "--help", 0

section .data
    numbers db 10, 20, 30
    count   equ 3

    signed_byte db -5
    float_val   dd 2.5

section .bss
    shared_mem resd 1

section .text
    extern printf
    extern puts
    extern fprintf
    extern strcmp
    extern exit
    extern stderr

    global main

init_func:
    push rbp
    mov rbp, rsp
    leave
    ret

fini_func:
    push rbp
    mov rbp, rsp
    leave
    ret

print_help_and_exit:
    push rdi                     ; save exit code
    lea rdi, [usage_fmt]
    mov rsi, [r15]               ; argv[0]
    call printf wrt ..plt
    lea rdi, [options_line]
    call puts wrt ..plt
    lea rdi, [help_option_line]
    call puts wrt ..plt
    pop rdi
    call exit wrt ..plt

main:
    mov r14, rdi                 ; argc
    mov r15, rsi                 ; argv

    cmp r14, 1
    je .start_processing

    cmp r14, 2
    jne .invalid_args

    mov rdi, [r15 + 8]           ; argv[1]
    lea rsi, [help_short]
    call strcmp wrt ..plt
    test rax, rax
    jz .show_help

    mov rdi, [r15 + 8]
    lea rsi, [help_long]
    call strcmp wrt ..plt
    test rax, rax
    jnz .invalid_args

.show_help:
    xor edi, edi                 ; exit code 0
    call print_help_and_exit

.invalid_args:
    mov rdi, [rel stderr]
    lea rsi, [invalid_msg]
    mov rdx, [r15 + 8]           ; the invalid argument
    call fprintf wrt ..plt
    mov rdi, 1                   ; exit code 1
    call print_help_and_exit

.start_processing:
    lea rdi, [status_msg]
    call puts wrt ..plt

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
    jl error_exit

    add eax, ebx

    lea r8, [do_exit]
    jmp r8

error_exit:
    mov eax, 98
    lea r8, [do_exit]
    jmp r8

do_exit:
    mov rdi, rax
    call exit wrt ..plt

; --- Subroutine ---
double_val:
    mov rax, rdi
    add rax, rax
    ret
