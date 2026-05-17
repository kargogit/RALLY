#include <stdio.h>
#include <unistd.h>
int main(void) {
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL)
        printf("%s\n", cwd);
    return 0;
}
