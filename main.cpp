#include <iostream>
#include "Solution.h"
#include "FreqStack.h"

TreeNode* BuildTree(vector<int> nums)
{
    TreeNode *root = new TreeNode(nums[0]);
    deque<TreeNode*> parents;
    parents.push_back(root);
    deque<TreeNode*> children;
    TreeNode * parent;
    bool left = true;
    for (int i = 1; i < nums.size(); i++) {
        if (left) {
            parent = parents.front();
            parents.pop_front();
        }
        if (nums[i] != INT32_MIN) {
            TreeNode *new_node = new TreeNode(nums[i]);
            if (left)
                parent->left = new_node;
            else
                parent->right = new_node;
            children.push_back(new_node);
        }
        left = !left;
        if (left && parents.empty()) {
            parents = children;
            children.clear();
        }
    }
    parent = nullptr;
    delete parent;
    return root;
}

int main() {
    Solution solution;

    vector<int> nums = {1,2};
    TreeNode* root = BuildTree(nums);
    solution.flatten(root);
    return 0;
}