//
// Created by zeal4u on 2019/4/20.
//

#include "TreeNode.h"
#include "binary_tree_problems.h"

int main() {
    TreeNode *root = TreeNode::BuildTree({6,1,12,0,3,10,13,null,null,null,null,4,14,20,16,2,5,11,15});
    std::list<int> result;
    binary_tree_problem::MorrisTraversal(binary_tree_problem::GetMaxBST(root),
            result, binary_tree_problem::INORDER);
    std::for_each(result.begin(), result.end(), [](int i) {printf("%d->", i);});
    printf("#\n");
    return 0;
}

