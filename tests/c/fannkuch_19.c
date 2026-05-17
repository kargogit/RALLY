#include <stdio.h>
#define LIKELY(x) __builtin_expect(!!(x), 1)
int main() {
    int val = 1;
    if (LIKELY(val)) printf("Likely branch taken\n");
}
