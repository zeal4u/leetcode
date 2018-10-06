//
// Created by zeal4u on 2018/9/17.
//

#ifndef LEETCODE_SOLUTION_H
#define LEETCODE_SOLUTION_H

#include <string>
#include <iostream>
#include <vector>
#include <deque>
#include <map>
#include <set>
#include <algorithm>
#include <cstdlib>
#include <time.h>
#include <cmath>
#include <stdlib.h>

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
        ListNode *result = &result_obj;
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

    // problem 714
    int maxProfit(vector<int> &prices, int fee) {
        int buy = INT32_MIN, sell = 0;
        for (int price:prices) {
            int sell_old = sell;
            sell = max(sell, buy + price);
            buy = max(buy, sell_old - price - fee);
        }
        return sell;
    }

    // problem 80
    int removeDuplicates(vector<int> &nums) {
        if (nums.size() <= 2)
            return nums.size();
        int count = 1, last_value = nums[0];
        for (auto item = nums.begin() + 1; item != nums.end();) {
            if (*item == last_value) {
                if (count == 0) {
                    item = nums.erase(item);
                    continue;
                } else {
                    count--;
                }
            } else {
                last_value = *item;
                count = 1;
            }
            item++;
        }
        return nums.size();
    }

    // problem 172
    int trailingZeroes(int n) {
        int result = 0;

        for (long long i = 5; n / i > 0; i *= 5) {
            result += (n / i);
        }
        return result;
    }

    // problem 728
    vector<int> selfDividingNumbers(int left, int right) {
        vector<int> result;
        int tmp = 0, digit = 0;
        bool flag;
        for (int i = left; i <= right; ++i) {
            flag = true;
            tmp = i;
            while (tmp > 0) {
                digit = tmp % 10;
                if (digit == 0 || i % digit != 0) {
                    flag = false;
                    break;
                }
                tmp = tmp / 10;
            }
            if (flag)
                result.push_back(i);
        }
        return result;
    }

    // problem 875
    int minEatingSpeed(vector<int> &piles, int H) {
        int low = 1, high = *max_element(piles.begin(), piles.end());
        while (low < high) {
            int middle = (low + high) / 2;
            int total_times = 0;
            for (int i:piles) {
                total_times += ceil(i * 1.0 / middle);
            }
            if (total_times > H)
                low = middle + 1;
            else
                high = middle;
        }
        return low;
    }

    // problem 862
    int shortestSubarray(vector<int> &A, int K) {
        int n = A.size();
        int res = n + 1;
        vector<int> B(n + 1, 0);
        for (int i = 0; i < n; ++i)
            B[i + 1] = B[i] + A[i];

        deque<int> d;
        for (int i = 0; i <= n; ++i) {
            while (!d.empty() && B[i] - B[d.front()] >= K) {
                res = min(res, i - d.front());
                d.pop_front();
            }
            while (!d.empty() && B[i] <= B[d.back()]) {
                d.pop_back();
            }
            d.push_back(i);
        }
        return res <= n ? res : -1;
    }

    // problem 763
    vector<int> partitionLabels(string S) {
        vector<vector<int>> char_maps(26, vector<int>());
        for (int i = 0; i < S.length(); ++i) {
            char_maps[S[i] - 'a'].push_back(i);
        }

        vector<int> result;
        int end = -1, cur = 0, start = 0, char_i;
        while (cur < S.length()) {
            char_i = S[cur] - 'a';
            if (end == -1) {
                start = cur;
                end = char_maps[char_i].back();
            }
            if (end == cur) {
                result.push_back(end - start + 1);
                end = -1;
            } else {
                // update end if new_end > end
                end = char_maps[char_i].back() > end ? char_maps[char_i].back() : end;
            }
            cur++;
        }
        return result;
    }

    TreeNode *tree_copy(TreeNode *pNode) {
        if (pNode == nullptr)
            return nullptr;
        TreeNode *root = new TreeNode(pNode->val);
        root->left = tree_copy(pNode->left);
        root->right = tree_copy(pNode->right);
        return root;
    }

    // problem 894
    vector<TreeNode *> allPossibleFBT(int N) {
        if ((N & 1) == 0)
            return vector<TreeNode *>();
        map<int, vector<TreeNode *>> component;
        component[1] = vector<TreeNode *>(1, new TreeNode(0));
        for (int cur_n = 3; cur_n <= N; cur_n += 2) {
            component[cur_n] = vector<TreeNode *>();
            for (int left_node_n = 1; left_node_n < cur_n; left_node_n += 2) {
                int right_node_n = cur_n - 1 - left_node_n;
                for (auto left_node:component[left_node_n]) {
                    for (auto right_node:component[right_node_n]) {
                        TreeNode *root = new TreeNode(0);
                        root->left = tree_copy(left_node);
                        root->right = tree_copy(right_node);
                        component[cur_n].push_back(root);
                    }
                }
            }
        }
        return component[2 * (N - 1) / 2 + 1];
    }

    // problem 883
    int projectionArea(vector<vector<int>> &grid) {
        if (grid.empty())
            return 0;
        int xy = 0, yz = 0, xz = 0;
        int max_row = 0, max_col[grid[0].size()];
        for (int i = 0; i < grid[0].size(); ++i)
            max_col[i] = 0;

        for (int i = 0; i < grid.size(); ++i) {
            max_row = 0;
            for (int j = 0; j < grid[0].size(); ++j) {
                if (grid[i][j] != 0)
                    xy++;
                if (max_row < grid[i][j]) {
                    max_row = grid[i][j];
                }
                if (max_col[j] < grid[i][j]) {
                    max_col[j] = grid[i][j];
                }
            }
            yz += max_row;
        }
        for (int i: max_col)
            xz += i;
        return xy + yz + xz;
    }

    // problem 2
    ListNode *addTwoNumbers(ListNode *l1, ListNode *l2) {
        ListNode *result = new ListNode(0);
        ListNode *pNode = result;
        while (l1 && l2) {
            pNode->next = new ListNode(0);
            pNode = pNode->next;

            pNode->val = l1->val + l2->val;

            l1 = l1->next;
            l2 = l2->next;
        }
        if (l1) {
            pNode->next = l1;
        }
        if (l2) {
            pNode->next = l2;
        }
        pNode = result;
        while (pNode) {
            if (pNode->val >= 10) {
                pNode->val = pNode->val - 10;
                if (pNode->next)
                    pNode->next->val++;
                else {
                    pNode->next = new ListNode(1);
                }
            }
            pNode = pNode->next;
        }
        ListNode *ans = result->next;
        delete result;
        return ans;
    }

    // problem 701
    TreeNode *insertIntoBST(TreeNode *root, int val) {
        if (!root)
            return new TreeNode(val);

        TreeNode *result = tree_copy(root);
        TreeNode *pNode = result;

        bool flag = true;
        while (flag) {
            if (pNode->val > val) {
                if (pNode->left) {
                    pNode = pNode->left;
                } else {
                    pNode->left = new TreeNode(val);
                    flag = false;
                }
            } else {
                if (pNode->right) {
                    pNode = pNode->right;
                } else {
                    pNode->right = new TreeNode(val);
                    flag = false;
                }
            }
        }
        return result;
    }

    // problem 867
    vector<vector<int>> transpose(vector<vector<int>> &A) {
        vector<vector<int>> B;
        if (A.empty())
            return B;
        for (int j = 0; j < A[0].size(); ++j) {
            B.emplace_back();
            for (auto &i : A) {
                B[j].push_back(i[j]);
            }
        }
        return B;
    }

    // problem 908
    int smallestRangeI(vector<int> &A, int K) {
        int A_max = -1, A_min = 10000;
        for (int i:A) {
            if (A_max < i)
                A_max = i;
            if (A_min > i)
                A_min = i;
        }
        return A_max - A_min - 2 * K < 0 ? 0 : A_max - A_min - 2 * K;
    }

    // problem 876
    ListNode *middleNode(ListNode *head) {
        ListNode *slow = head, *fast = head;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
        }
        return slow;
    }

    // problem 885
    vector<vector<int>> spiralMatrixIII(int R, int C, int r0, int c0) {
        vector<vector<int>> result(R * C, vector<int>(2, 0));
        int direction = 0, radius = 1, steps = 0;
        bool direction_change = false;
        for (int i = 0;;) {
            if (r0 >= 0 && r0 < R && c0 >= 0 && c0 < C) {
                // record legal pos
                result[i][0] = r0, result[i][1] = c0;
                i++;
            }
            if (i == R * C)
                break;
            switch (direction) {
                case 0:
                    c0++;
                    if (direction_change) {
                        radius++;
                        direction_change = false;
                    }
                    break;
                case 1:
                    r0++;
                    break;
                case 2:
                    c0--;
                    if (direction_change) {
                        radius++;
                        direction_change = false;
                    }
                    break;
                case 3:
                    r0--;
                    break;
            }
            steps++;
            if (steps == radius) {
                // change direction
                direction = (direction + 1) % 4;
                steps = 0;
                direction_change = true;
            }

        }
        return result;
    }

    // problem 700
    TreeNode *searchBST(TreeNode *root, int val) {
        TreeNode *pNode = root;
        while (pNode && pNode->val != val) {
            if (pNode->val < val)
                pNode = pNode->right;
            else
                pNode = pNode->left;
        }
        return pNode;
    }

    // problem 806
    vector<int> numberOfLines(vector<int> &widths, string S) {
        int space = 100;
        int count = 1;
        for (char c:S) {
            if (space - widths[c - 'a'] < 0) {
                count++;
                space = 100 - widths[c - 'a'];
            } else {
                space -= widths[c - 'a'];
            }
        }
        vector<int> result(2, 0);
        result[0] = count, result[1] = 100 - space;
        return result;
    }

    // problem 559
    int maxDepth(Node *root) {
        if (!root)
            return 0;
        int res = 1;
        for (Node *ch:root->children) {
            res = max(res, maxDepth(ch) + 1);
        }
        return res;
    }

    // problem 811
    vector<string> subdomainVisits(vector<string> &cpdomains) {
        map<string, int> count;
        for (string &s:cpdomains) {
            for (int i = 0; i < s.length(); ++i) {
                if (s[i] == ' ') {
                    string domain = s.substr(i + 1, s.length() - 1 - i);
                    int n = stoi(s.substr(0, i));
                    count[domain] += n;
                    for (int j = 0; j < domain.length(); ++j) {
                        if (domain[j] == '.') {
                            string sub_d = domain.substr(j + 1, domain.length() - 1 - j);
                            count[sub_d] += n;
                        }
                    }
                    break;
                }
            }
        }
        vector<string> result;
        for (auto &pair:count) {
            result.push_back(to_string(pair.second) + " " + pair.first);
        }
        return result;
    }

    // problem 589
    void _preorder(vector<int> &v, Node *pNode) {
        if (pNode) {
            v.push_back(pNode->val);
            for (Node *child:pNode->children) {
                _preorder(v, child);
            }
        }
    }

    // problem 589
    vector<int> preorder(Node *root) {
        vector<int> result;
        _preorder(result, root);
        return result;
    }

    // problem 821
    vector<int> shortestToChar(string S, char C) {
        vector<int> result;
        vector<int> locate;
        int n = S.length(), min_d = n;
        for (int i = 0; i < n; ++i) {
            if (S[i] == C)
                locate.push_back(i);
        }
        for (int i = 0; i < n; ++i) {
            min_d = n;
            for (int l: locate) {
                if (min_d > abs(l - i)) {
                    min_d = abs(l - i);
                    if (min_d == 0)
                        break;
                }
            }
            result.push_back(min_d);
        }
        return result;
    }

    // problem 791
    struct my_cmp{
        string cmp_str;
        my_cmp(string str):cmp_str(str){}
        bool operator()(char i, char j){
            int ii = cmp_str.find(i);
            int jj = cmp_str.find(j);
            return ii < jj;
        }
    };

    // problem 791
    string customSortString(string S, string T) {
        struct my_cmp my_cmp1(S);
        sort(T.begin(), T.end(), my_cmp1);
        return T;
    }

    // problem 872
    bool leafSimilar(TreeNode* root1, TreeNode* root2) {
        vector<int> leafs1, leafs2;
        _leafSimilar(leafs1, root1);
        _leafSimilar(leafs2, root2);
        if(leafs1.size() ==  leafs2.size())
            for(int i=0;i<leafs1.size();++i){
                if(leafs1[i] != leafs2[i])
                    return false;
            }
        else{
            return false;
        }
        return true;
    }

    // problem 872
    void _leafSimilar(vector<int> &leafs, TreeNode * root){
        if(root){
            if(!(root->left || root->right))
                leafs.push_back(root->val);
            else{
                _leafSimilar(leafs, root->left);
                _leafSimilar(leafs, root->right);
            }
        }
    }

    // problem 884
    vector<string> uncommonFromSentences(string A, string B) {
        int a_n = A.size(), b_n = B.size();
        int a_index=0, b_index=0;
        map<string, int> count;
        for(int i=0; i<a_n;++i){
            if(A[i] == ' '){
                count[A.substr(a_index, i - a_index)] += 1;
                a_index = i+1;
            }
        }
        count[A.substr(a_index, a_n - a_index)] += 1;
        for(int i=0; i<b_n;++i){
            if(B[i] == ' '){
                count[B.substr(b_index, i - b_index)] += 1;
                b_index = i+1;
            }
        }
        count[B.substr(b_index, b_n - b_index)] += 1;

        vector<string> result;
        for(auto &pair:count){
            if(pair.second == 1)
                result.push_back(pair.first);
        }
        return result;
    }


    // problem 766
    bool isToeplitzMatrix(vector<vector<int>>& matrix) {
        int row = matrix.size();
        int col = matrix[0].size();
        for(int i=0;i<row;++i)
            for(int j = 0;j< col;++j){
                if(i+1<row && j+1<col && matrix[i][j] != matrix[i+1][j+1])
                    return false;
            }
        return true;
    }

    // problem 868
    int binaryGap(int N) {
        int gap=0, start=-1, end=0, tmp;
        for(int i=0;i<31;++i){
            if(((N>>i)&1) == 1){
                end = i;
                tmp = end - start;
                if(gap < tmp && start != -1){
                    gap = tmp;
                }
                start = end;
            }
        }
        return gap;
    }
};

#endif //LEETCODE_SOLUTION_H
