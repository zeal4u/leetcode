//
// Created by zeal4u on 2018/9/17.
//

#ifndef LEETCODE_SOLUTION_H
#define LEETCODE_SOLUTION_H

#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>

#include "Employee.h"
#include "Node.h"
#include "TreeNode.h"

using namespace std;

class Solution {
public:
    // problem 771
    int numJewelsInStones(string J, string S) {
        int result = 0;
        for (char i : J) {
            for (char j : S)
                if (i == j)
                    result++;
        }
        return result;
    }

    // problem 807
    int maxIncreaseKeepingSkyline(vector<vector<int>> &grid) {
        if (grid.size() == 0)
            exit(-1);
        unsigned int col_n = grid[0].size();
        unsigned int row_n = grid.size();

        // record highest number of cols
        vector<int> top_skyline(col_n, 0);
        // record highest number of rows
        vector<int> left_skyline(row_n, 0);

        // find highest skyline through top or left
        for (int i = 0; i < row_n; i++) {
            for (int j = 0; j < col_n; j++) {
                if (grid[i][j] > top_skyline[j]) {
                    top_skyline[j] = grid[i][j];
                }
                if (grid[i][j] > left_skyline[i]) {
                    left_skyline[i] = grid[i][j];
                }
            }
        }

        int sum = 0;
        int distance = 0;
        // compute distances between skyline and real heights
        for (int i = 0; i < row_n; i++) {
            for (int j = 0; j < col_n; j++) {
                if (top_skyline[j] <= left_skyline[i]) {
                    sum += top_skyline[j] - grid[i][j];
                } else {
                    sum += left_skyline[i] - grid[i][j];
                }
            }
        }
        return sum;
    }

    // problem 905
    vector<int> sortArrayByParity(vector<int> &A) {
        vector<int> result(A.size(), 0);

        int k = 0, j = A.size() - 1;
        for (int i : A) {
            if ((i & 1) == 0) {
                result[k] = i;
                k++;
            } else {
                result[j] = i;
                j--;
            }
        }
        return result;
    }

    // problem 709
    string toLowerCase(string str) {
        string result = "";
        for (char i : str) {
            if (i >= 65 && i <= 90) {
                result += (i + 32);
            } else {
                result += i;
            }
        }
        return result;
    }

    // problem 804
    int uniqueMorseRepresentations(vector<string> &words) {
        string code_list[26] = {".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..",
                                "--", "-.",
                                "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."};
        vector<string> codes;
        for (int i = 0; i < words.size(); i++) {
            string word = words[i];
            string code = "";
            for (char j : word) {
                code += code_list[j - 'a'];
            }
            bool flag = true;
            for (int k = 0; k < codes.size(); k++) {
                if (code.compare(codes[k]) == 0) {
                    flag = false;
                    break;
                }
            }
            if (flag)
                codes.push_back(code);
        }
        return codes.size();
    }

    // problem 890
    vector<string> findAndReplacePattern(vector<string> &words, string pattern) {
        map<char, vector<int>> pattern_map;
        pattern_map[pattern[0]] = vector<int>(1, 0);
        // divide into same char sets
        for (int i = 1; i < pattern.length(); i++) {
            auto item = pattern_map.find(pattern[i]);
            if (item != pattern_map.end())
                (*item).second.push_back(i);
            else
                pattern_map[pattern[i]] = vector<int>(1, i);
        }

        vector<string> result;
        // check that 1)chars in same set must be the same, 2)chars in diff sets must be different.
        for (auto &word : words) {
            bool flag = true;
            string record;
            for (auto &pair : pattern_map) {
                vector<int> same_list = pair.second;
                // the second rule
                if (record.find(word[same_list[0]]) == string::npos)
                    record += word[same_list[0]];
                else {
                    flag = false;
                }

                if (!flag)
                    break;

                // the first rule
                for (int i = 1; i < same_list.size(); ++i) {
                    if (word[same_list[i]] != word[same_list[0]]) {
                        flag = false;
                        break;
                    }
                }

            }

            if (flag)
                result.push_back(word);
        }
        return result;
    }

    // problem 832
    vector<vector<int>> flipAndInvertImage(vector<vector<int>> &A) {
        if (A.size() == 0)
            exit(-1);
        vector<vector<int>> result;
        for (int i = 0; i < A.size(); ++i) {
            result.push_back(vector<int>());
            for (int j = A[0].size() - 1; j >= 0; --j) {
                result[i].push_back(A[i][j] ^ 1);
            }
        }
        return result;
    }

    // problem 690
    int getImportance(vector<Employee *> employees, int id) {
        vector<int> cur_sub_ids(1, id);
        vector<int> nex_sub_ids;

        int sum = 0;
        while (!cur_sub_ids.empty()) {
            for (auto item = employees.begin(); item != employees.end();) {
                bool is_erase = false;
                for (int i = 0; i < cur_sub_ids.size(); ++i) {
                    if ((*item)->id == cur_sub_ids[i]) {
                        sum += (*item)->importance;
                        nex_sub_ids.insert(nex_sub_ids.end(), (*item)->subordinates.begin(),
                                           (*item)->subordinates.end());
                        item = employees.erase(item);
                        is_erase = true;
                        break;
                    }
                }
                if (!is_erase)
                    ++item;
            }
            cur_sub_ids.clear();
            cur_sub_ids = nex_sub_ids;
            nex_sub_ids.clear();
        }
        return sum;
    }

    // problem 590
    // Recursive solution is trivial, could you do it iteratively?
    vector<int> postorder_rec(Node *root) {
        vector<int> result;
        if (root != nullptr) {
            vector<Node *> &children = root->children;
            for (auto item = children.begin(); item != children.end(); ++item) {
                vector<int> tmp = postorder_rec(*item);
                result.insert(result.end(), tmp.begin(), tmp.end());
            }
            result.push_back(root->val);
        }
        return result;
    }

    // problem 590
    vector<int> postorder(Node *root) {
        vector<int> result;
        if (root != nullptr) {
            // has or not detected children
            set<Node *> detected_node;
            // front is top of the stack
            vector<Node *> stack;
            stack.push_back(root);
            while (!stack.empty()) {
                if (detected_node.end() == detected_node.find(stack.front())) {
                    // new hot node
                    vector<Node *> &children = stack.front()->children;
                    detected_node.insert(stack.front());
                    if (!children.empty()) {
                        stack.insert(stack.begin(), children.begin(), children.end());
                    } else {
                        // get no children, record its value, then kick it out
                        result.push_back(stack.front()->val);
                        stack.erase(stack.begin());
                    }
                } else {
                    // already detect children
                    result.push_back(stack.front()->val);
                    stack.erase(stack.begin());
                }

            }
        }
        return result;
    }

    // 674
    int findLengthOfLCIS(vector<int> &nums) {
        int result = 0;
        int cur_inc_len = 1;
        for (int i = 1; i < nums.size(); ++i) {
            if (nums[i] <= nums[i - 1]) {
                if (cur_inc_len > result)
                    result = cur_inc_len;
                cur_inc_len = 1;
            } else
                cur_inc_len++;
        }
        if (nums.size() != 0 && cur_inc_len > result)
            result = cur_inc_len;
        return result;
    }

    // problem 645
    vector<int> findErrorNums(vector<int> &nums) {
        if (nums.empty())
            return vector<int>();
        vector<int> result(2, 0);
        vector<int> count(nums.size(), 0);
        for (int i = 0; i < nums.size(); ++i) {
            result[1] ^= (i + 1) ^ nums[i];
            if (++count[nums[i] - 1] == 2)
                result[0] = nums[i];
        }
        result[1] = result[1] ^ result[0];
        return result;
    }

    // problem 814
    TreeNode *pruneTree(TreeNode *root) {
        if (root != nullptr) {
            root->left = pruneTree(root->left);
            root->right = pruneTree(root->right);

            if (root->val == 0 && root->right == nullptr && root->left == nullptr) {
                root = nullptr;
            }
        }
        return root;
    }

    // problem 852
    int peakIndexInMountainArray(vector<int> &A) {
        int index = 0;

        if (A.size() <= 1) {
            //pass
        } else if (A.size() <= 3) {
            for (int i = 1; i < A.size(); ++i) {
                if (A[i] < A[i - 1]) {
                    index = i - 1;
                    break;
                } else {
                    index = i;
                }
            }
        } else {
            int left = A.size() / 4;
            int right = 3 * A.size() / 4;
            vector<int> B;
            if (A[left] > A[right]) {
                B.insert(B.end(), A.begin(), A.begin() + right);
                index = peakIndexInMountainArray(B);
            } else {
                B.insert(B.end(), A.begin() + left, A.end());
                index = left + peakIndexInMountainArray(B);
            }
        }
        return index;
    }

    // problem 516
    int longestPalindromeSubseq(string s) {
        int len = s.size();
        vector<vector<int>> storage(len, vector<int>(len));

        for (int i = 0; i < len; ++i) storage[i][i] = 1;
        for (int k = 1; k < len; ++k) {
            for (int j = 0; j < len - k; ++j) {
                storage[j][j + k] = s[j] == s[j + k] ? 2 + storage[j + 1][j + k - 1] : max(storage[j + 1][j + k],
                                                                                           storage[j][j + k - 1]);
            }
        }
        return storage[0][len - 1];
    }

    // problem 861
    int matrixScore(vector<vector<int>> &A) {
        vector<vector<int>> B = A;

        // 1. toggle any row to make 1s are at the highest digit.
        for (int i = 0; i < B.size(); ++i) {
            if (B[i][0] == 0) {
                for (int j = 0; j < B[i].size(); ++j) {
                    B[i][j] ^= 1;
                }
            }
        }
        // 2. toggle any column to make 1s more than 0s
        for (int j = 1; j < B[0].size(); ++j) {
            int count = 0;
            for (int i = 0; i < B.size(); ++i) {
                if (B[i][j] == 0)
                    count++;

            }
            if (count > B.size() / 2) {
                for (int i = 0; i < B.size(); ++i) {
                    B[i][j] ^= 1;
                }
            }
        }

        int sum = 0;
        int power = 1;
        for (int i = B[0].size() - 1; i >= 0; --i) {
            for (int j = 0; j < B.size(); ++j) {
                sum += B[j][i] * power;
            }
            power *= 2;
        }
        return sum;

    }

    // problem 797
    vector<vector<int>> allPathsSourceTarget(vector<vector<int>> &graph) {
        int n = graph.size();
        vector<vector<int>> routes;

        vector<int> cur_route;
        int cur_node = 0;
        visitGraph(graph, routes, cur_route, cur_node);
        return routes;
    }

    // problem 797
    void visitGraph(vector<vector<int>> &graph, vector<vector<int>> &routes, vector<int> cur_route, int cur_node) {
        cur_route.push_back(cur_node);
        if (cur_node == graph.size() - 1) {
            routes.push_back(cur_route);
        }
        for (int i: graph[cur_node])
            visitGraph(graph, routes, cur_route, i);
    }

    // problem 190
    uint32_t reverseBits(uint32_t n) {
        uint32_t result = 0;
        for (int i = 0; i < 32; ++i) {
            result |= ((n >> i) & 1) << (31 - i);
        }
        return result;
    }

    // problem 740
    int deleteAndEarn(vector<int> &nums) {
        if (nums.empty())
            return 0;
        int n = nums.size();
        sort(nums.begin(), nums.end());

        vector<int> max_sum(n + 1, 0);
        max_sum[1] = nums[0];
        // if pick num[i], then every value equals to nums[i]-1 and nums[i]+1 needs to be deleted
        for (int i = 2; i <= n; ++i) {
            // find nums having same value
            int j = i - 1;
            for (; j > 0; --j) {
                // diff value needs deleting.
                if (nums[j - 1] == nums[i - 1] - 1) {
                    int k = j - 1;
                    // compute number of value equals to nums[i] - 1
                    for (; k > 0; --k) {
                        if (nums[j - 1] != nums[k - 1]) {
                            break;
                        }
                    }
                    max_sum[i] = max(max_sum[j], max_sum[k] + (i - j) * nums[i - 1]);
                    break;
                } else if (nums[j - 1] != nums[i - 1]) {
                    max_sum[i] = max_sum[j] + (i - j) * nums[i - 1];
                    break;
                }
            }
            if (j == 0) {
                max_sum[i] = max_sum[j] + (i - j) * nums[i - 1];
            }
        }
        return max_sum[n];
    }

    // problem 21
    struct ListNode {
        int val;
        ListNode *next;

        ListNode(int x) : val(x), next(NULL) {}
    };

    ListNode *mergeTwoLists(ListNode *l1, ListNode *l2) {
        if (l1 == nullptr)
            return l2;
        if (l2 == nullptr)
            return l1;
        ListNode result_obj = ListNode(0);
        ListNode* result = & result_obj;
        ListNode *p_node = result;
        while (l1 && l2) {
            if (l1->val > l2->val) {
                p_node->next = l2;
                l2 = l2->next;
            } else {
                p_node->next = l1;
                l1 = l1->next;
            }
            p_node = p_node->next;
        }
        while (l1) {
            p_node->next = l1;
            l1 = l1->next;
            p_node = p_node->next;
        }
        while (l2) {
            p_node->next = l2;
            l2 = l2->next;
            p_node = p_node->next;
        }
        return result->next;
    }
};

#endif //LEETCODE_SOLUTION_H
