section .data
    status_msg db "Processing...", 10, 0          ; Null-terminate for printf
    msg_len equ $ - status_msg

    numbers db 10, 20, 30
    count   equ 3

    signed_byte db -5
    float_val   dd 2.5

    usage_msg db "Usage: program [options]", 10
              db "Options:", 10
              db "  -h    Show help message", 10, 0
    error_msg db "Error: Invalid operation detected", 10, 0

section .init_array
    dq constructor_stub      ; Constructor pointer

section .fini_array
    dq destructor_stub       ; Destructor pointer

section .bss
    shared_mem resd 1

section .text
    default rel              ; Enable RIP-relative addressing globally

    extern printf, fprintf, exit, stderr
    global main

; --- Standard ELF constructor (runs before main) ---
constructor_stub:
    ret                      ; Minimal valid constructor

; --- Standard ELF destructor (runs after main) ---
destructor_stub:
    ret                      ; Minimal valid destructor

; --- Dedicated help function (prints multi-line usage and exits) ---
print_help:
    lea rdi, [rel usage_msg]
    xor eax, eax             ; Clear AL for variadic call
    call printf
    mov rdi, 0               ; Exit code 0
    call exit                ; Uses PLT indirection

; --- Error reporter (prints to stderr using PLT) ---
report_error:
    lea rdi, [rel stderr wrt ..got]  ; RIP-relative GOT access for stderr
    mov rdi, [rdi]                   ; Dereference to get FILE* for stderr
    lea rsi, [rel error_msg]
    xor eax, eax
    call fprintf            ; PLT call to libc
    mov rdi, 98
    call exit               ; PLT call

; --- Original subroutine (unchanged) ---
double_val:
    mov rax, rdi
    add rax, rax
    ret

; --- Main entry (replaces _start; processes args, uses libc) ---
main:
    push rbp
    mov rbp, rsp

    ; --- Process command-line arguments (argc in RDI, argv in RSI) ---
    cmp rdi, 2               ; Check if at least 1 arg provided
    jl .skip_arg_check
    mov rax, [rsi + 8]       ; argv[1]
    cmp byte [rax], '-'
    jne .skip_arg_check
    cmp byte [rax + 1], 'h'
    jne .skip_arg_check
    call print_help          ; Handles -h option (exits internally)

.skip_arg_check:
    ; --- Print status using libc (PLT call) ---
    lea rdi, [rel status_msg]
    xor eax, eax
    call printf              ; PLT indirection for external call

    ; --- Computational core (minimal changes from original) ---
    mov rbx, -1
    add rbx, 1
    setc bl
    movzx eax, bl

    lea rsi, [numbers]
    mov rcx, count

.sum_loop:
    add al, [rsi]
    test al, al
    setpe dl
    lea r8, [.even_parity]
    lea r9, [.next_iter]
    test dl, dl
    cmove r8, r9
    jmp r8

.even_parity:
    add al, 1
    jmp .next_iter

.next_iter:
    inc rsi
    loop .sum_loop

    shl eax, 2
    movsx ecx, byte [signed_byte]
    imul ecx, 2
    lea edx, [rax + rcx]     ; Fixed register usage (RAX holds shifted value)

    movss xmm0, [float_val]
    addss xmm0, xmm0
    cvttss2si eax, xmm0
    add edx, eax

    mov rdi, rdx
    call double_val          ; Internal call (no PLT needed)

    mov edi, 100
    xchg dword [shared_mem], edi
    lock inc dword [shared_mem]

    mov ecx, 10
    cdq
    idiv ecx

    cmp edx, 4
    setne dl

    mov eax, 0x7FFFFFFF
    add eax, 1
    jo .overflow_ok

    mov eax, 99
    jmp .comp_do_exit        ; Direct jump to local exit handler

.overflow_ok:
    mov eax, 1
    add eax, 44
    cmp eax, -1
    jl .comp_error_exit      ; Conditional branch to error path
    add eax, ebx
    jmp .comp_do_exit

.comp_error_exit:            ; Local label (avoids symbol conflict)
    jmp report_error         ; Branch to error handler (uses PLT internally)

.comp_do_exit:               ; Local label
    mov rdi, rax             ; Exit code in RDI
    call exit                ; PLT call for teardown (replaces syscall)

    ; Note: C runtime handles final teardown via .fini_array
