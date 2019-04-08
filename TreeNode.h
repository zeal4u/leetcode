//
// Created by zeal4u on 2018/9/20.
//

#ifndef LEETCODE_TREENODE_H
#define LEETCODE_TREENODE_H

#include <vector>
#include <deque>
#include <stack>
#include <stdint-gcc.h>

#define null INT32_MIN


class TreeNode {
 public:
  int val;
  TreeNode *left;
  TreeNode *right;

  TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}

  static TreeNode *BuildTree(std::vector<int> nums);

  static void test_tree();
};


extern void pre_order(TreeNode *root, std::vector<TreeNode *> &result);

extern void in_order(TreeNode *root, std::vector<TreeNode *> &result);

#endif //LEETCODE_TREENODE_H
