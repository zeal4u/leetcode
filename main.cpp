#include <iostream>
#include "Solution.h"
#include "FreqStack.h"

int main() {
    Solution solution;

    TreeNode* root  = TreeNode::BuildTree({10,5,-3,3,2,null,11,3,-2,null,1});

    cout<<solution.rob(root);

    return 0;
}