#include <signal.h>
#include <string.h>
#include <stdio.h>
int main(void) {
  printf("Signal 9: %s\n", strsignal(SIGKILL));
  return 0;
}
