#include <stdio.h>
int main() {
    unsigned long long counter = 0;
    unsigned long long old = __sync_fetch_and_add(&counter, 100);
    printf("Old: %llu, New: %llu\n", old, counter);
}
