section .text
global _start

_start:
    mov edi, 10         ; Pass an argument (in EDI for simplicity)
    call get_double     ; Call the function
    ; After ret, RAX should hold 20
    mov edi, eax        ; Move result to exit status
    mov eax, 60         ; sys_exit
    syscall

get_double:
    ; Function: doubles the value in EDI and returns it in RAX
    mov eax, edi        ; Move argument to EAX
    add eax, eax        ; Double it
    ret                 ; Return to caller
