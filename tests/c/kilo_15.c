#include <stdlib.h>
#include <string.h>
#include <unistd.h>
typedef struct { char *b; int len; } abuf;
void abAppend(abuf *ab, const char *s, int len) {
    ab->b = realloc(ab->b, ab->len + len);
    memcpy(ab->b + ab->len, s, len);
    ab->len += len;
}
int main() { return 0; }
