//
// Created by zeal4u on 2019/3/4.
//
#include "TreeNode.h"

#include <iostream>
#include <algorithm>

void TreeNode::test_tree() {
  //TreeNode* root  = TreeNode::BuildTree({10,5,-3,3,2,null,11,3,-2,null,1,12});
  //TreeNode* root  = TreeNode::BuildTree({});
  //TreeNode* root  = TreeNode::BuildTree({10, 5, null, 3, null, 3});
  TreeNode *root = TreeNode::BuildTree({10, null, 5, null, 3, null, 4});

  std::vector<TreeNode *> result;
  pre_order(root, result);
  std::for_each(result.begin(), result.end(), [](TreeNode *tn) { std::cout << tn->val << " "; });
  std::cout << std::endl;
  result.clear();
  in_order(root, result);
  std::for_each(result.begin(), result.end(), [](TreeNode *tn) { std::cout << tn->val << " "; });
}

void pre_order(TreeNode *root, std::vector<TreeNode *> &result) {
  if (nullptr == root)
    return;
  std::stack<TreeNode *> s;
  TreeNode *node = nullptr;
  while (nullptr != root || !s.empty()) {
    while (nullptr != root) {
      result.push_back(root);
      s.push(root);
      root = root->left;
    }
    node = s.top();
    s.pop();
    if (nullptr != node->right)
      root = node->right;
  }
}

void in_order(TreeNode *root, std::vector<TreeNode *> &result) {
  if (nullptr == root)
    return;
  std::stack<TreeNode *> s;
  TreeNode *node = nullptr;
  while (nullptr != root || !s.empty()) {
    while (nullptr != root) {
      s.push(root);
      root = root->left;
    }
    node = s.top();
    s.pop();
    result.push_back(node);
    if (nullptr != node->right)
      root = node->right;
  }
}

TreeNode *TreeNode::BuildTree(std::vector<int> nums) {
  if (0 == nums.size() || null == nums[0])
    return nullptr;
  TreeNode *root = new TreeNode(nums[0]);
  std::deque<TreeNode *> parents;
  parents.push_back(root);
  std::deque<TreeNode *> children;
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