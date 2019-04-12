#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string.h>
#include <stack>
#include "TreeNode.h"

using namespace std;
bool isValidRect(vector<vector<char>> &board, int count[], int index) {
    int i_start = (index / 3) * 3, j_start = (index % 3) * 3, i_range = i_start + 3, j_range = j_start + 3;
    for (int i = i_start; i < i_range; i++) {
        for (int j = j_start; j < j_range; j++) {
            if (board[i][j] != '.') {
                count[board[i][j] - '1']++;
                if (count[board[i][j] - '1'] >= 2)
                    return false;
            }
        }
    }
    return true;
}
bool isValidSudoku(vector<vector<char>>& board) {
    int count[9] = {0};
    for (int i = 0 ; i < 9; i++) {
        memset(count, 0, sizeof(count));
        for (int j = 0; j < 9; j++) {
            if (board[i][j] != '.') {
                count[board[i][j] - '1']++;
                if (count[board[i][j] - '1'] >= 2)
                    return false;
            }
        }
    }

    for (int i = 0 ; i < 9; i++) {
        memset(count, 0, sizeof(count));
        for (int j = 0; j < 9; j++) {
            if (board[j][i] != '.') {
                count[board[j][i] - '1']++;
                if (count[board[j][i] - '1'] >= 2)
                    return false;
            }
        }
    }

    bool res = true;
    for (int i = 0; i < 9; i++) {
        memset(count, 0, sizeof(count));
        res &= isValidRect(board, count, i);
        if (!res)
            return false;
    }
    return true;
}

vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
    if (root == nullptr)
        return {};
    stack<TreeNode*> left, right;
    vector<vector<int>> result;
    left.push(root);
    bool to_left = true;
    while (!left.empty() || !right.empty()) {
        if (to_left) {
            vector<int> tmp;
            while (!left.empty()) {
                TreeNode *node = left.top();
                left.pop();

                tmp.push_back(node->val);
                if (node->left) {
                    right.push(node->left);
                }
                if (node->right) {
                    right.push(node->right);
                }

            }
            result.push_back(move(tmp));
        } else {
            vector<int> tmp;
            while (!right.empty()) {
                TreeNode *node = right.top();
                right.pop();

                tmp.push_back(node->val);
                if (node->right) {
                    left.push(node->right);
                }
                if (node->left) {
                    left.push(node->left);
                }
            }
            result.push_back(move(tmp));
        }
        to_left = ~to_left;
    }
    return result;
}
int main()
{
    zigzagLevelOrder(TreeNode::BuildTree({3,9,20,null,null,15,7}));
    return 0;
}
