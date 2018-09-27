#include <iostream>
#include "Solution.h"

int main() {
    Solution solution;
    Solution::ListNode l_11(1);
    Solution::ListNode l_12(2);
    Solution::ListNode l_13(4);
    l_11.next = &l_12;
    l_12.next = &l_13;
    Solution::ListNode* l1 = &l_11;

    Solution::ListNode l_21(1);
    Solution::ListNode l_22(3);
    Solution::ListNode l_23(4);
    l_21.next = &l_22;
    l_22.next = &l_23;
    Solution::ListNode* l2 = &l_21;

    auto result = solution.mergeTwoLists(l1, l2);
    while(result){
        cout<<result->val<<" ";
        result = result->next;
    }
    return 0;
}