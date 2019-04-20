//
// Created by zeal4u on 2019/4/20.
//

#ifndef LEETCODE_BINARY_TREE_PROBLEMS_H
#define LEETCODE_BINARY_TREE_PROBLEMS_H

#include <stdio.h>
#include <list>
#include <unordered_map>
#include <algorithm>
#include "TreeNode.h"


namespace binary_tree_problem {
const static int PREORDER = 1;
const static int INORDER = 2;

    // Morris Traversal of binary tree
    // Leverage nullptr of leaf to go back to upper level of tree
    // order PREORDER
    // order INORDER
    void MorrisTraversal(TreeNode *root, std::list<int> &storage, int order) {
        TreeNode *cur_node = root;
        while (cur_node) {
            if (cur_node->left == nullptr) {
                storage.push_back(cur_node->val);
                cur_node = cur_node->right;
            } else {
                TreeNode *most_right = cur_node->left;
                while (most_right->right && most_right->right != cur_node) {
                    most_right = most_right->right;
                }
                if (most_right->right == nullptr) {
                    if (order == PREORDER)
                        storage.push_back(cur_node->val);
                    most_right->right = cur_node;
                    cur_node = cur_node->left;
                } else {
                    if (order == INORDER)
                        storage.push_back(cur_node->val);
                    most_right->right = nullptr;
                    cur_node = cur_node->right;
                }
            }
        }
    }

    // Assistant method
    // save the sum from root to current node, then cur_sum - sum must be some pre sum if exists
    int PreOrderFind(
            TreeNode *root,
            std::unordered_map<int, int>& record,
            int max_len,
            int sum, int pre_sum, int level) {
        if (root == nullptr)
            return max_len;
        int cur_sum = pre_sum + root->val;
        if (record.find(cur_sum) == record.end())
            record[cur_sum] = level;
        if (record.find(cur_sum - sum) != record.end()) {
            max_len = std::max(max_len, level - record[cur_sum - sum]);
        }

        max_len = PreOrderFind(root->left, record, max_len, sum, cur_sum, level + 1);
        max_len = PreOrderFind(root->right, record, max_len, sum, cur_sum, level + 1);
        if (level == record[cur_sum])
            record.erase(cur_sum);
        return max_len;
    }

    // Find the longest path given sum of a binary Tree
    // Path is route from up to down
    // require Method: PreOrderFind
    int FindLongestPathWithSum(TreeNode *root, int sum) {
        std::unordered_map<int, int> record;
        record[0] = 0;
        return PreOrderFind(root, record, 0, sum, 0, 1);
    }

    // Find the largest binary tree in a normal binary tree
    struct ReturnType {
        TreeNode *root_;
        int max_bst_size_;
        int min_val_;
        int max_val_;
        ReturnType(TreeNode *root, int max_bst_szie, int min_val, int max_val) :
            root_(root), max_bst_size_(max_bst_szie), min_val_(min_val), max_val_(max_val) {}
    };

    ReturnType process(TreeNode *root) {
        if (root == nullptr)
            return ReturnType(nullptr, 0, INT32_MAX, INT32_MIN);
        ReturnType left_rt = process(root->left);
        ReturnType right_rt = process(root->right);
        int min_val = std::min({left_rt.min_val_, right_rt.min_val_, root->val});
        int max_val = std::max({left_rt.max_val_, right_rt.max_val_, root->val});
        int max_bst_size = std::max(left_rt.max_bst_size_, right_rt.max_bst_size_);
        TreeNode *head = max_bst_size == left_rt.max_bst_size_ ? left_rt.root_ : right_rt.root_;
        if (root->left == left_rt.root_ &&
            root->right == right_rt.root_ &&
            root->val >= left_rt.max_val_ &&
            root->val <= right_rt.min_val_) {
            max_bst_size = left_rt.max_bst_size_ + right_rt.max_bst_size_ + 1;
            head = root;
        }
        return ReturnType(head, max_bst_size, min_val, max_val);
    }
    // Find the largest binary tree in a normal binary tree
    // main method
    TreeNode* GetMaxBST(TreeNode *root) {
        return process(root).root_;
    }

} // namespace binary_tree_problem

#endif //LEETCODE_BINARY_TREE_PROBLEMS_H
