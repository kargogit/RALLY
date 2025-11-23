section .text
global _start

_start:
    mov eax, 5          ; first value
    cmp eax, 6          ; compare with second value
    jz  equal           ; jump if equal
    mov edi, 1          ; not equal → exit status 1
    jmp done

equal:
    mov edi, 0          ; equal → exit status 0

done:
    mov eax, 60         ; sys_exit
    syscall
