section .text
global _start

_start:
    movsx   rax, byte [value]   ; sign-extend byte to qword
    add     rax, [counter]      ; add to counter
    mov     [counter], rax      ; store result back (though not strictly needed)
    mov     rdi, rax            ; exit status = sum
    mov     rax, 60             ; sys_exit
    syscall

section .data
    value   db -5               ; example signed byte
    counter dq 10               ; example qword counter
