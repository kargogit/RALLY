;---------------------------------------------------------------------------
;  Combined test – contains all the features from the 28 separate programs
;  Assemble:  nasm -felf64 combined.asm -o combined.o
;  Link:      ld -o combined combined.o
;  Run:       ./combined
;---------------------------------------------------------------------------

section .data
    ; data needed by the various sub‑tests
    numbers      db 3, 1, 4, 1, 5, 9          ; array for sum test
    msg          db "Hello World!", 10      ; string for strlen test
    flt1         dd 2.5                     ; float operands
    flt2         dd 1.5
    flt3         dd 2.0
    yolo         db 0xAB                    ; byte for nibble‑mask test
    value        db -5                      ; signed byte for sign‑extend test
    counter64    dq 10                      ; qword for sign‑extend test
    outstr       db "All tests done", 10    ; message we write to stdout

section .bss
    counter      resb 1                     ; byte‑size mutable variable
    shared       resd 1                     ; 32‑bit word for atomic xchg
    buffer       resb 16                    ; buffer for sys_read

section .text
global _start

_start:
    ;-------------------------------------------------------------------
    ; 1. shift – logical (SHR) vs arithmetic (SAR) on a negative number
    ;-------------------------------------------------------------------
    mov  eax, -8
    mov  ebx, eax
    shr  eax, 1
    sar  ebx, 1
    shl  ebx, 8
    or   eax, ebx                ; result is discarded

    ;-------------------------------------------------------------------
    ; 2. parity adjust – test + jpe (parity‑even jump)
    ;-------------------------------------------------------------------
    mov  eax, 5
    test eax, eax
    jpe  .skip_parity
    add  eax, 1
.skip_parity:

    ;-------------------------------------------------------------------
    ; 3. LEA – compute address expression
    ;-------------------------------------------------------------------
    lea  rax, [rax + rax*2 + 3]   ; rax ← rax + rax*2 + 3

    ;-------------------------------------------------------------------
    ; 4. BSS write – modify a global byte
    ;-------------------------------------------------------------------
    mov  byte [counter], 42
    movzx edi, byte [counter]    ; result stored (will be overwritten later)

    ;-------------------------------------------------------------------
    ; 5. array sum – sum a small byte array
    ;-------------------------------------------------------------------
    lea  rsi, [numbers]
    mov  ecx, 6
    xor  eax, eax
.sum_loop:
    add  al, [rsi]
    inc  rsi
    loop .sum_loop
    movzx edi, al                ; sum = 23 → 0x17 (overwritten)

    ;-------------------------------------------------------------------
    ; 6. strlen – simple null‑terminated string length
    ;-------------------------------------------------------------------
    lea  rsi, [msg]
    xor  ecx, ecx
.strlen_loop:
    cmp  byte [rsi + rcx], 0
    je   .strlen_done
    inc  ecx
    jmp  .strlen_loop
.strlen_done:                    ; length = 13 (overwritten)

    ;-------------------------------------------------------------------
    ; 7. stack operations – push / pop
    ;-------------------------------------------------------------------
    mov  rax, 0x1111111111111111
    mov  rbx, 0x2222222222222222
    push rax
    push rbx
    pop  rdi                    ; now rdi = 0x2222...

    ;-------------------------------------------------------------------
    ; 8. signed multiply (imul) – 64‑bit result in rdx:rax
    ;-------------------------------------------------------------------
    mov  rax, -10
    mov  rdx, 5
    imul rdx                    ; rdx:rax = -50

    ;-------------------------------------------------------------------
    ; 9. simple function call / return
    ;-------------------------------------------------------------------
    mov  edi, 10
    call get_double             ; returns 20 in eax

    ;-------------------------------------------------------------------
    ;10. signed vs unsigned comparison (jl / jb)
    ;-------------------------------------------------------------------
    mov  eax, -1
    cmp  eax, 1
    jl   .signed_less
    jb   .unsigned_below
    mov  edi, 99                 ; none taken
    jmp  .cmp_done
.signed_less:
    jb   .error_jb
    mov  edi, 1                  ; signed < true, unsigned not
    jmp  .cmp_done
.unsigned_below:
    mov  edi, 98
    jmp  .cmp_done
.error_jb:
    mov  edi, 98
.cmp_done:

    ;-------------------------------------------------------------------
    ;11. sys_write – write our status line to stdout
    ;-------------------------------------------------------------------
    mov  rax, 1                  ; sys_write
    mov  rdi, 1                  ; stdout
    lea  rsi, [outstr]
    mov  rdx, 15                 ; length of "All tests done\n"
    syscall

    ;-------------------------------------------------------------------
    ;12. division (div) – 17 / 5 → quotient=3, remainder=2
    ;-------------------------------------------------------------------
    xor  edx, edx
    mov  eax, 17
    mov  ecx, 5
    div  ecx                     ; eax = 3, edx = 2

    ;-------------------------------------------------------------------
    ;13. overflow flag detection (add causing signed overflow)
    ;-------------------------------------------------------------------
    mov  eax, 0x7FFFFFFF
    add  eax, 1
    jo   .overflow
    mov  edi, 0
    jmp  .overflow_done
.overflow:
    mov  edi, 1
.overflow_done:

    ;-------------------------------------------------------------------
    ;14. sys_read – read up to 16 bytes (will return 0 on EOF)
    ;-------------------------------------------------------------------
    mov  rax, 0                  ; sys_read
    xor  rdi, rdi                ; stdin
    lea  rsi, [buffer]
    mov  rdx, 16
    syscall

    ;-------------------------------------------------------------------
    ;15. lock xchg – atomic exchange with a memory location
    ;-------------------------------------------------------------------
    mov  eax, 42
    lock xchg [shared], eax      ; eax = old value (0)

    ;-------------------------------------------------------------------
    ;16. atomic inc – lock inc on a byte
    ;-------------------------------------------------------------------
    mov  byte [counter], 0
    lock inc byte [counter]
    lock inc byte [counter]
    movzx edi, byte [counter]    ; counter = 2

    ;-------------------------------------------------------------------
    ;17. floating‑point arithmetic (SSEx)
    ;-------------------------------------------------------------------
    movss xmm0, [flt1]           ; 2.5
    addss xmm0, [flt2]           ; +1.5 → 4.0
    mulss xmm0, [flt3]           ; *2.0 → 8.0
    cvttss2si edi, xmm0          ; truncate to integer → 8

    ;-------------------------------------------------------------------
    ;18. add with carry – setc after adding -1 + 1
    ;-------------------------------------------------------------------
    mov  rax, -1
    add  rax, 1                  ; result 0, CF = 1
    setc al
    movzx edi, al                ; edi = 1

    ;-------------------------------------------------------------------
    ;19. simple add constant
    ;-------------------------------------------------------------------
    mov  eax, 5
    add  eax, 3
    mov  edi, eax                ; edi = 8

    ;-------------------------------------------------------------------
    ;20. inc – increment a register
    ;-------------------------------------------------------------------
    xor  eax, eax
    inc  eax
    mov  edi, eax                ; edi = 1

    ;-------------------------------------------------------------------
    ;21. cmp + setne – set byte on “not equal”
    ;-------------------------------------------------------------------
    mov  eax, 5
    cmp  eax, 6
    setne al
    movzx edi, al                ; edi = 1

    ;-------------------------------------------------------------------
    ;22. cmp + conditional branch (jz)
    ;-------------------------------------------------------------------
    mov  eax, 5
    cmp  eax, 6
    jz   .equal
    mov  edi, 1
    jmp  .branch_done
.equal:
    mov  edi, 0
.branch_done:

    ;-------------------------------------------------------------------
    ;23. bitwise chain – and / or / xor
    ;-------------------------------------------------------------------
    mov  eax, 0x0F
    and  eax, 0x03               ; 0x03
    or   eax, 0x08               ; 0x0B
    xor  eax, 0x01               ; 0x0A (10)
    mov  edi, eax

    ;-------------------------------------------------------------------
    ;24. nibble mask – keep low 4 bits
    ;-------------------------------------------------------------------
    movzx eax, byte [yolo]
    and  eax, 0x0F               ; 0x0B & 0x0F = 0x0B (11)
    mov  edi, eax

    ;-------------------------------------------------------------------
    ;25. sign‑extend + add – signed byte + qword
    ;-------------------------------------------------------------------
    movsx rax, byte [value]      ; -5
    add  rax, [counter64]       ; +10 → 5
    mov  rdi, rax                ; final exit code = 5

    ;-------------------------------------------------------------------
    ; exit
    ;-------------------------------------------------------------------
    mov  eax, 60                 ; sys_exit
    syscall

;-------------------------------------------------------------------
; helper function used by the “call” test
;-------------------------------------------------------------------
get_double:
    mov  eax, edi
    add  eax, eax                ; double the argument
    ret
