#include <iostream>
#include "Solution.h"
#include "FreqStack.h"


int main() {
    Solution solution;

    vector<int> nums = {1,2};
    TreeNode* root = TreeNode::BuildTree(nums);
    solution.flatten(root);
    return 0;
}