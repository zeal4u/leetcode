#include <iostream>
#include "Solution.h"
#include "FreqStack.h"

int main() {
    Solution solution;

    vector<int> nums = {3,9,20,null, null, 15, 7};
    vector<int> preorder = {3, 9, 20, 15, 7};
    vector<int> inorder = {9,3,15,20,7};
    TreeNode* root = TreeNode::BuildTree(nums);
    solution.buildTree(preorder, inorder);
    return 0;
}