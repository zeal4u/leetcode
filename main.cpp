#include <iostream>
#include "Solution.h"
#include "FreqStack.h"

int main() {
    Solution solution;

    TreeNode * n1 = new TreeNode(4);
    TreeNode * n2 = new TreeNode(2);
    TreeNode * n3 = new TreeNode(6);
    TreeNode * n4 = new TreeNode(3);
    TreeNode * n5 = new TreeNode(1);
    TreeNode * n6 = new TreeNode(5);
    n1->left = n2;
    n1->right = n3;
    n2->left = n4;
    n2->right = n5;
    n3->left = n6;
    solution.addOneRow(n1, 1, 2);
    return 0;
}