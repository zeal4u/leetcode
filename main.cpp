#include <iostream>
#include "Solution.h"
#include "FreqStack.h"

int main() {
    Solution solution;

    ListNode *a1 = new ListNode(1);
    ListNode *a2 = new ListNode(1);
    ListNode *a3 = new ListNode(1);
    ListNode *a4 = new ListNode(1);
    a1->next = a2, a2->next = a3, a3->next = a4;
    solution.splitListToParts(a1, 5);
    return 0;
}