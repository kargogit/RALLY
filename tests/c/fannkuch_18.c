#include <sys/stat.h>
#include <stdio.h>
int main(void) {
    struct stat dot, root;
    if (stat(".", &dot) == 0 && stat("/", &root) == 0)
        puts(dot.st_ino == root.st_ino ? "At Root" : "Not Root");
    return 0;
}
