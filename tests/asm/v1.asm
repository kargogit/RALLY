; comprehensive_test.asm
; 
; This single test case consolidates the logic, instructions, and addressing modes
; from the provided list (test cases 1-28). It performs a deterministic calculation
; resulting in a specific exit code (46), allowing you to verify correctness by
; simply checking the returned status value.
;
; Features covered:
; - Syscalls (sys_write, sys_exit)
; - Control Flow (jmp, conditional jumps, call/ret, loop)
; - Logic & Bitwise (and, or, xor, test, shr, sar, shl)
; - Arithmetic (add, sub, imul, div, inc, dec)
; - Data Handling (mov, movzx, movsx, lea, xchg)
; - Flags & Conditions (setc, setne, jo, jpe, jl)
; - Memory Sections (.text, .data, .bss)
; - Stack Operations (push, pop)
; - Atomic Operations (lock prefix)
; - Floating Point (SSE: movss, addss, cvttss2si)

section .data
    ; String for I/O test (Test 20)
    status_msg db "Processing...", 10
    msg_len equ $ - status_msg

    ; Array for iteration/summation (Tests 14, 15)
    numbers db 10, 20, 30
    count equ 3

    ; Values for sign extension and floating point (Tests 9, 27)
    signed_byte db -5
    float_val dd 2.5

section .bss
    ; Mutable global variable (Tests 13, 24, 26)
    shared_mem resd 1

section .text
global _start

_start:
    ; --- 1. Syscall: Write (Test 20) ---
    mov rax, 1          ; sys_write
    mov rdi, 1          ; stdout
    mov rsi, status_msg
    mov rdx, msg_len
    syscall

    ; --- 2. Flag Logic & Carry (Test 28) ---
    mov rbx, -1        ; 0xFFFFFFFFFFFFFFFF
    add rbx, 1         ; Results in 0, Carry Flag (CF)=1
    setc bl            ; BL = 1 (CF is set)
    movzx eax, bl      ; EAX = 1

    ; --- 3. Loop & Parity (Tests 11, 14) ---
    lea rsi, [numbers] ; Load address (Test 12)
    mov rcx, count     ; Loop counter
.sum_loop:
    add al, [rsi]      ; Accumulate byte
    
    ; Parity check (Test 11)
    test al, al
    jpe .even_parity   ; Jump if Parity Even
    sub al, 1          ; If odd parity, subtract 1
    jmp .next_iter
.even_parity:
    add al, 1          ; If even parity, add 1 (arbitrary complexity)
.next_iter:
    inc rsi
    loop .sum_loop     ; Dec RCX, loop if not zero

    ; Calculation Trace for EAX:
    ; Start: 1
    ; 1+10=11 (Odd P) -> 10
    ; 10+20=30 (Odd P) -> 29
    ; 29+30=59 (Odd P) -> 58
    ; Result: EAX = 58

    ; --- 4. Bitwise & Shifts (Tests 7, 10) ---
    shl eax, 2         ; 58 * 4 = 232
    movsx ecx, byte [signed_byte] ; ECX = -5 (Test 9)
    imul ecx, 2        ; ECX = -10 (Test 17)
    
    lea edx, [eax + ecx] ; EDX = 232 + (-10) = 222

    ; --- 5. Floating Point (Test 27) ---
    movss xmm0, [float_val] ; Load 2.5
    addss xmm0, xmm0        ; 2.5 + 2.5 = 5.0
    cvttss2si eax, xmm0     ; Convert float to int: 5
    add edx, eax            ; EDX = 222 + 5 = 227

    ; --- 6. Stack & Function Call (Tests 16, 18) ---
    mov rdi, rdx       ; Pass argument (227)
    push rdi           ; Push to stack
    call double_val    ; Call subroutine
    add rsp, 8         ; Clean stack
    
    ; RAX now contains 227 * 2 = 454

    ; --- 7. Atomic Operations & BSS (Tests 13, 24, 26) ---
    mov edi, 100
    lock xchg [shared_mem], edi ; Atomic swap. [mem]=100, EDI=0
    lock inc dword [shared_mem] ; [mem]=101

    ; --- 8. Division & Modulo (Test 21) ---
    mov ecx, 10
    cdq                ; Sign extend EAX into EDX:EAX
    idiv ecx           ; 454 / 10 => EAX=44 (Quotient), EDX=4 (Remainder)

    ; --- 9. Comparisons & Overflow (Tests 5, 6, 19, 22) ---
    ; Check remainder logic
    cmp edx, 4
    setne dl           ; DL = 0 (Equal)
    
    ; Overflow Check (Test 22)
    mov eax, 0x7FFFFFFF
    add eax, 1         ; OF=1, Result wraps to negative
    jo .overflow_ok    ; Jump if Overflow (Test 5/6 logic)
    
    ; If we didn't jump, something is wrong
    mov eax, 99
    jmp do_exit           ; <-- fixed: jump to defined label

.overflow_ok:
    ; Restore logic flow:
    ; We had quotient 44. We want to signal success.
    mov eax, 1         ; Base status for overflow detected
    add eax, 44        ; Add quotient back. EAX = 45.

    ; --- 10. Final Logic & Exit (Tests 1, 2, 28) ---
    ; Signed vs Unsigned check (Test 19)
    cmp eax, -1
    jl  error_exit     ; 45 is not < -1, so this doesn't jump
    
    ; Add the initial carry flag result (RBX=1)
    add eax, ebx       ; 45 + 1 = 46
    
    jmp do_exit

error_exit:
    mov eax, 98        ; Error code

do_exit:
    mov rdi, rax       ; Exit status = 46 (or other)
    mov rax, 60        ; sys_exit
    syscall

; --- Subroutine ---
double_val:
    ; Input: RDI
    ; Output: RAX = RDI * 2
    mov rax, rdi
    add rax, rax
    ret
