//
// Created by zeal4u on 2018/9/20.
//

#ifndef LEETCODE_TREENODE_H
#define LEETCODE_TREENODE_H

#define null INT32_MIN

class TreeNode {
public:
    int val;
    TreeNode *left;
    TreeNode *right;

    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}

    static int test_val;

    static TreeNode *BuildTree(vector<int> nums) {
        TreeNode *root = new TreeNode(nums[0]);
        deque<TreeNode *> parents;
        parents.push_back(root);
        deque<TreeNode *> children;
        TreeNode *parent;
        bool left = true;
        for (int i = 1; i < nums.size(); i++) {
            if (left) {
                parent = parents.front();
                parents.pop_front();
            }
            if (nums[i] != null) {
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
        return root;
    }

};


#endif //LEETCODE_TREENODE_H
