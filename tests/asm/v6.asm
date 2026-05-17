; minimal changes to add: init/fini arrays, libc calls (PLT), default rel, main, print_help
default rel

section .data
    status_msg      db "Processing...", 10, 0
    msg_len         equ $ - status_msg - 1

    numbers         db 10, 20, 30, 0
    count           equ 3

    signed_byte     db -5
    float_val       dd 2.5

    usage_msg       db "Usage: prog [options]", 10
                    db "  -h, --help    Show this help and exit", 10, 0

    help_opt1       db "-h", 0
    help_opt2       db "--help", 0

    init_msg        db "init_func running (constructor).", 10, 0
    fini_msg        db "fini_func running (destructor).", 10, 0

section .bss
    shared_mem      resd 1

section .init_array
    dq init_func

section .fini_array
    dq fini_func

section .text
global main
extern printf
extern fprintf
extern strcmp
extern exit
extern stderr

; -------- init / fini --------
init_func:
    push rbp
    mov rbp, rsp
    lea rdi, [rel init_msg]
    xor eax, eax        ; no SSE regs used for variadic call
    call printf
    pop rbp
    ret

fini_func:
    push rbp
    mov rbp, rsp
    lea rdi, [rel fini_msg]
    xor eax, eax
    call printf
    pop rbp
    ret

; -------- main & argument handling --------
; int main(int argc, char **argv)
main:
    push rbp
    mov rbp, rsp

    ; argc in edi, argv in rsi
    cmp edi, 2
    jl .do_process

    ; check argv[1] == "-h"
    mov rax, [rsi + 8]      ; argv[1]
    mov rdi, rax
    lea rsi, [rel help_opt1]
    call strcmp
    test rax, rax
    je .call_help

    ; check argv[1] == "--help"
    mov rax, [rsi + 8]      ; argv[1] (reload, rsi unchanged by call)
    mov rdi, rax
    lea rsi, [rel help_opt2]
    call strcmp
    test rax, rax
    je .call_help

    jmp .do_process

.call_help:
    call print_help        ; print_help will call exit

.do_process:
    call run_processing   ; returns status in eax
    mov edi, eax
    call exit             ; go through libc (PLT)

    ; never reached
    pop rbp
    ret

; -------- print_help (prints multi-line usage text and exits) --------
print_help:
    ; fprintf(stderr, usage_msg)
    push rbp
    mov rbp, rsp

    ; load FILE* stderr into rdi
    mov rax, [rel stderr]
    mov rdi, rax
    lea rsi, [rel usage_msg]
    xor eax, eax
    call fprintf

    ; exit(0)
    mov edi, 0
    call exit

    pop rbp
    ret

; -------- processing function (your original program adapted to return an int in EAX) --------
run_processing:
    push rbp
    mov rbp, rsp

    ; --- 1. Print using libc instead of raw syscall ---
    lea rdi, [rel status_msg]
    xor eax, eax
    call printf

    ; --- 2. Flag Logic & Carry (Test 28) ---
    mov rbx, -1
    add rbx, 1
    setc bl
    movzx eax, bl

    ; --- 3. Loop & Parity (Tests 11, 14) ---
    lea rsi, [rel numbers]
    mov rcx, count         ; loop counter (RCX must survive parity test)
    xor al, al             ; clear AL accumulation for safety

.sum_loop:
    add al, [rsi]          ; accumulate byte in AL

    ; Parity check with no clobber of RCX
    test al, al
    setpe dl               ; use DL (not CL) so RCX is preserved
    lea r8, [rel .even_parity]
    lea r9, [rel .next_iter]
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
    mov eax, 0             ; ensure EAX known before shifts
    mov eax, eax           ; no-op preserving EAX state
    shl eax, 2
    movsx ecx, byte [rel signed_byte]
    imul ecx, 2
    lea edx, [eax + ecx]

    ; --- 5. Floating Point (Test 27) ---
    movss xmm0, [rel float_val]
    addss xmm0, xmm0
    cvttss2si eax, xmm0
    add edx, eax

    ; --- 6. Stack & Function Call (Tests 16, 18) ---
    mov rdi, rdx           ; pass via rdi (no push/pop, preserves alignment)
    call double_val

    ; return value in RAX from double_val, but keep mixing with edx as before
    ; rax holds doubled value; combine with edx
    add edx, eax

    ; --- 7. Atomic Operations & BSS (Tests 13, 24, 26) ---
    mov edi, 100
    xchg dword [rel shared_mem], edi   ; xchg is atomic; no need for LOCK prefix
    lock inc dword [rel shared_mem]

    ; --- 8. Division & Modulo (Test 21) ---
    mov ecx, 10
    mov eax, edx          ; dividend lower
    cdq
    idiv ecx

    ; --- 9. Comparisons & Overflow (Tests 5, 6, 19, 22) ---
    cmp edx, 4
    setne dl

    mov eax, 0x7FFFFFFF
    add eax, 1
    jo .overflow_ok

    mov eax, 99
    jmp .do_exit

.overflow_ok:
    mov eax, 1
    add eax, 44

    cmp eax, -1
    jl  .error_exit

    add eax, ebx

    jmp .do_exit

.error_exit:
    mov eax, 98
    jmp .do_exit

.do_exit:
    ; return value in EAX
    pop rbp
    ret

; --- Subroutine ---
double_val:
    ; expects argument in RDI, returns in RAX
    push rbp
    mov rbp, rsp
    mov rax, rdi
    add rax, rax
    pop rbp
    ret
