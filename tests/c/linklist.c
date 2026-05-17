#include <stdio.h>
#include <stdlib.h>

struct Node { int data; struct Node* next; };

int main() {
    struct Node* head = malloc(sizeof(struct Node));
    head->data = 1;
    head->next = malloc(sizeof(struct Node));
    head->next->data = 2;
    head->next->next = NULL;

    for(struct Node* n = head; n; n = n->next)
        printf("%d ", n->data);

    return 0;
}
