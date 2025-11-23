section .text
global _start

_start:
    mov eax, -1
    cmp eax, 1

    jl  is_less_signed   ; Jump if less (signed). This should jump.
    jb  is_below_unsigned ; Jump if below (unsigned). This should NOT jump.

    ; If jl failed to jump, we exit with error code 99
    mov edi, 99
    jmp done

is_less_signed:
    ; If jb incorrectly jumps, we exit with error code 98
    jb  error_jb
    ; Correct path: jl jumped, jb did not jump
    mov edi, 1
    jmp done

error_jb:
    mov edi, 98

done:
    mov eax, 60 ; sys_exit
    syscall
