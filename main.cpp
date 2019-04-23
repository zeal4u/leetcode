//
// Created by zeal4u on 2019/4/20.
//

#include "TreeNode.h"
#include "binary_tree_problems.h"
#include "util.h"

void TestMorrisTraveral() {
  TreeNode *root = TreeNode::BuildTree({6, 1, 12, 0, 3, 10, 13, null, null, null, null, 4, 14, 20, 16, 2, 5, 11, 15});
  std::list<int> result;
  binary_tree_problem::MorrisTraversal(binary_tree_problem::GetMaxBST(root),
                                       result, binary_tree_problem::INORDER);
  std::for_each(result.begin(), result.end(), [](int i) { printf("%d->", i); });
  printf("#\n");
}

void TestBSTTopoSize() {
  TreeNode *root = TreeNode::BuildTree({6, 1, 12, 0, 3, 10, 13, null, null, null, null, 4, 14, 20, 16, 2, 5, 11, 15});
  int result = binary_tree_problem::BSTTopoSize(root);
  printf("%d\n", result);
}

void TestIsSubStr() {
  int res = 0;
  res = tools::IsSubStr("abcdb", "bc");
  printf("%d\n", res);

  res = tools::IsSubStr("abcdb", "bce");
  printf("%d\n", res);

  res = tools::IsSubStr("aaaaaa", "a");
  printf("%d\n", res);
}

void TestIsSubTree() {
  TreeNode *root1 = TreeNode::BuildTree({1, 2, 3, 4, 5, 6, 7});
  TreeNode *root2 = TreeNode::BuildTree({2, 4, 5});
  printf("%d\n", binary_tree_problem::IsSubTree(root1, root2));
}

int main(int argc, char *argv[]) {
  return 0;
}


