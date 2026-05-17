#include <stdio.h>

struct Point {
    int x;
    int y;
};

int main(void) {
    struct Point p = {3, 7};        // create and initialise a struct
    printf("%d %d\n", p.x, p.y);     // access its members
    return 0;
}
