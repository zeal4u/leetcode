//
// Created by zeal4u on 2018/9/20.
//

#ifndef LEETCODE_TREENODE_H
#define LEETCODE_TREENODE_H


class TreeNode {
public:
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x):val(x),left(nullptr),right(nullptr){}
};


#endif //LEETCODE_TREENODE_H
