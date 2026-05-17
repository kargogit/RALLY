#include <unistd.h>
#include <stdio.h>
int main(void) {
    char buf[1024];
    if (getcwd(buf, sizeof(buf)))
        puts(buf);
    return 0;
}
