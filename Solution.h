//
// Created by zeal4u on 2018/9/17.
//

#ifndef LEETCODE_SOLUTION_H
#define LEETCODE_SOLUTION_H

#include <string>
#include <iostream>
#include <vector>
#include <queue>
#include <deque>
#include <map>
#include <set>
#include <algorithm>
#include <cstdlib>
#include <time.h>
#include <cmath>
#include <stdlib.h>
#include <stack>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <utility>

#include "Employee.h"
#include "Node.h"
#include "TreeNode.h"

using namespace std;


struct ListNode {
        int val;
        ListNode *next;

        ListNode(int x) : val(x), next(NULL) {}
};


class Solution {
public:
    Solution(){}
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
    struct my_cmp {
        string cmp_str;

        my_cmp(string str) : cmp_str(str) {}

        bool operator()(char i, char j) {
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
    bool leafSimilar(TreeNode *root1, TreeNode *root2) {
        vector<int> leafs1, leafs2;
        _leafSimilar(leafs1, root1);
        _leafSimilar(leafs2, root2);
        if (leafs1.size() == leafs2.size())
            for (int i = 0; i < leafs1.size(); ++i) {
                if (leafs1[i] != leafs2[i])
                    return false;
            }
        else {
            return false;
        }
        return true;
    }

    // problem 872
    void _leafSimilar(vector<int> &leafs, TreeNode *root) {
        if (root) {
            if (!(root->left || root->right))
                leafs.push_back(root->val);
            else {
                _leafSimilar(leafs, root->left);
                _leafSimilar(leafs, root->right);
            }
        }
    }

    // problem 884
    vector<string> uncommonFromSentences(string A, string B) {
        int a_n = A.size(), b_n = B.size();
        int a_index = 0, b_index = 0;
        map<string, int> count;
        for (int i = 0; i < a_n; ++i) {
            if (A[i] == ' ') {
                count[A.substr(a_index, i - a_index)] += 1;
                a_index = i + 1;
            }
        }
        count[A.substr(a_index, a_n - a_index)] += 1;
        for (int i = 0; i < b_n; ++i) {
            if (B[i] == ' ') {
                count[B.substr(b_index, i - b_index)] += 1;
                b_index = i + 1;
            }
        }
        count[B.substr(b_index, b_n - b_index)] += 1;

        vector<string> result;
        for (auto &pair:count) {
            if (pair.second == 1)
                result.push_back(pair.first);
        }
        return result;
    }


    // problem 766
    bool isToeplitzMatrix(vector<vector<int>> &matrix) {
        int row = matrix.size();
        int col = matrix[0].size();
        for (int i = 0; i < row; ++i)
            for (int j = 0; j < col; ++j) {
                if (i + 1 < row && j + 1 < col && matrix[i][j] != matrix[i + 1][j + 1])
                    return false;
            }
        return true;
    }

    // problem 868
    int binaryGap(int N) {
        int gap = 0, start = -1, end = 0, tmp;
        for (int i = 0; i < 31; ++i) {
            if (((N >> i) & 1) == 1) {
                end = i;
                tmp = end - start;
                if (gap < tmp && start != -1) {
                    gap = tmp;
                }
                start = end;
            }
        }
        return gap;
    }

    // problem 682
    int calPoints(vector<string> &ops) {
        int sum = 0, n;
        vector<int> integers;
        for (auto item = ops.begin(); item != ops.end(); ++item) {
            if (*item == "+") {
                n = integers.size();
                integers.push_back(integers[n - 1] + integers[n - 2]);
            } else if (*item == "D") {
                n = integers.size();
                integers.push_back(integers[n - 1] * 2);
            } else if (*item == "C") {
                integers.pop_back();
            } else {
                integers.push_back(stoi(*item));
            }
        }
        for (int i:integers)
            sum += i;
        return sum;
    }

    // problem 442
    vector<int> findDuplicates(vector<int> &nums) {
        vector<int> result;
        int n = nums.size();
        for (int i = 0; i < n; ++i) {
            nums[abs(nums[i]) - 1] = -nums[abs(nums[i]) - 1];
            if (nums[abs(nums[i]) - 1] > 0)
                // 取负翻转两次为正
                result.push_back(abs(nums[i]));
        }
        return result;
    }

    // problem 897
    TreeNode *increasingBST(TreeNode *root) {
        vector<int> in_order;
        _increasingBST(in_order, root);
        TreeNode *result = new TreeNode(0);
        TreeNode *pNode = result;
        for (int i: in_order) {
            pNode->right = new TreeNode(i);
            pNode = pNode->right;
        }
        return result->right;
    }

    // problem 897
    void _increasingBST(vector<int> &in_order, TreeNode *root) {
        if (root) {
            _increasingBST(in_order, root->left);
            in_order.push_back(root->val);
            _increasingBST(in_order, root->right);
        }
    }

    // problem 877
    bool stoneGame(vector<int> &piles) {
        return true;
    }

    // problem 841
    bool canVisitAllRooms(vector<vector<int>> &rooms) {
        set<int> all_keys = {0};
        set<int> cur_keys(rooms[0].begin(), rooms[0].end());
        set<int> next_keys;
        while (!cur_keys.empty()) {
            next_keys.clear();
            for (int i : cur_keys) {
                if (all_keys.find(i) == all_keys.end()) {
                    next_keys.insert(rooms[i].begin(), rooms[i].end());
                }
            }
            all_keys.insert(cur_keys.begin(), cur_keys.end());
            cur_keys = next_keys;
        }
        return all_keys.size() == rooms.size();
    }

    // problem 693
    bool hasAlternatingBits(int n) {
        long long i = (n & 1) == 1 ? 1 : 0;
        long long factor = i == 1 ? 4 : 2;
        while (i < n) {
            i += factor;
            factor *= 4;
        }
        return i == n;
    }

    // problem 540
    int singleNonDuplicate(vector<int> &nums) {
        int start = 0, end = nums.size() - 1;
        while (start < end) {
            int mid = (start + end) / 2;
            if (mid % 2 == 1) mid--;
            if (nums[mid] != nums[mid + 1]) end = mid;
            else start = mid + 2;
        }
        return nums[start];
    }

    // problem 739
    vector<int> dailyTemperatures(vector<int> &temperatures) {
        int n = temperatures.size();
        vector<int> result(n, 0);
        stack<int> s;
        for (int i = n - 1; i >= 0; --i) {
            while (!s.empty() && temperatures[i] >= temperatures[s.top()]) {
                s.pop();
            }
            result[i] = s.empty() ? 0 : s.top() - i;
            s.push(i);
        }
        return result;
    }

    // problem 429
    vector<vector<int>> levelOrder(Node *root) {
        vector<vector<int>> result;
        if (!root)
            return result;
        vector<Node *> cur_level = {root};
        vector<Node *> next_level;
        while (!cur_level.empty()) {
            next_level.clear();
            result.push_back(vector<int>());
            for (Node *n:cur_level) {
                if (n) {
                    next_level.insert(next_level.end(), n->children.begin(), n->children.end());
                    result.back().push_back(n->val);
                }
            }
            cur_level = next_level;
        }
        return result;
    }

    // problem 896
    bool isMonotonic(vector<int> &A) {
        int n = A.size();
        if (n <= 2)
            return true;
        int direction = A[1] - A[0], tmp;
        for (int i = 1; i < n - 1; ++i) {
            tmp = A[i + 1] - A[i];
            if (tmp == 0)
                continue;
            else if (tmp * direction < 0)
                return false;
            else
                direction = tmp;
        }
        return true;
    }

    // problem 889
    TreeNode *constructFromPrePost(vector<int> &pre, vector<int> &post) {
        vector<TreeNode *> s;
        TreeNode *node;
        s.push_back(new TreeNode(pre[0]));
        for (int i = 1, j = 0; i < pre.size(); ++i) {
            while (s.back()->val == post[j]) {
                s.pop_back();
                j++;
            }
            node = new TreeNode(pre[i]);
            if (s.back()->left == nullptr) {
                s.back()->left = node;
            } else {
                s.back()->right = node;
            }
            s.push_back(node);
        }
        return s[0];
    }

    // problem 892
    int surfaceArea(vector<vector<int>> &grid) {
        int col = grid[0].size();
        int row = grid.size();
        int result = 0;
        int front, back, left, right;
        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; ++j) {
                if (grid[i][j]) {
                    result += 2;
                    if (i - 1 >= 0) {
                        back = grid[i][j] - grid[i - 1][j];
                        result += back > 0 ? back : 0;
                    } else {
                        result += grid[i][j];
                    }
                    if (i + 1 < row) {
                        front = grid[i][j] - grid[i + 1][j];
                        result += front > 0 ? front : 0;
                    } else {
                        result += grid[i][j];
                    }
                    if (j - 1 >= 0) {

                        left = grid[i][j] - grid[i][j - 1];
                        result += left > 0 ? left : 0;
                    } else {
                        result += grid[i][j];
                    }
                    if (j + 1 < col) {
                        right = grid[i][j] - grid[i][j + 1];
                        result += right > 0 ? right : 0;
                    } else {
                        result += grid[i][j];
                    }
                }
            }
        }
        return result;
    }

    // problem 888
    vector<int> fairCandySwap(vector<int> &A, vector<int> &B) {
        vector<int> result(2, 0);
        int sum_a = accumulate(A.begin(), A.end(), 0), sum_b = accumulate(B.begin(), B.end(), 0);

        int diff = (sum_a - sum_b) / 2;
        for (int i:A)
            for (int j:B) {
                if (i - j == diff) {
                    result[0] = i;
                    result[1] = j;
                    return result;
                }
            }
        return result;
    }

    // problem 647
    int countSubstrings(string s) {
        int n = s.length(), result = n;
        vector<vector<bool>> dp(n, vector<bool>(n, 0));

        for (int i = 0; i < n; ++i) dp[i][i] = true;

        for (int i = 1; i < n; ++i) {
            for (int j = 0; j <= n - i - 1; ++j) {
                dp[j][j + i] = s[j] == s[j + i] && (j + 1 <= j + i - 1 ? dp[j + 1][j + i - 1] : true);
                result += dp[j][j + i];
            }
        }
        return result;
    }

    // problem 784
    vector<string> letterCasePermutation(string S) {
        vector<string> result = {S};
        int n = 0;
        for (int i = 0; i < S.length(); ++i) {
            if (S[i] >= '0' && S[i] <= '9')
                continue;
            n = result.size();
            for (int j=0;j<n;++j) {
                string s = result[j];
                if (s[i] >= 'a' && s[i] <= 'z') {
                    s[i] = s[i] - 'a' + 'A';
                } else {
                    s[i] = s[i] - 'A' + 'a';
                }
                result.push_back(s);
            }
        }
        return result;
    }

    // problem 812
    double largestTriangleArea(vector<vector<int>>& points) {
        double res = 0;
        for(auto &i:points)
            for(auto &j:points)
                for(auto &k:points)
                    res = max(res, 0.5*(i[0]*(j[1]-k[1])+ j[0]*(k[1]-i[1])+k[0]*(i[1]-j[1])));
        return res;
    }

    // problem 865
    int scoreOfParentheses(string S) {
        int result = 0;
        stack<char> sk;
        bool flag = true;
        for(char c:S){
            if(c == '(') {
                sk.push(c);
                flag = true;
            } else {
                if(flag)
                    result += pow(2,sk.size()-1);
                flag = false;
                sk.pop();
            }
        }
        return result;
    }

    // problem 695
    int maxAreaOfIsland(vector<vector<int>>& grid) {
        int result = 0, row = grid.size(), col = grid[0].size();
        vector<vector<bool>> flags(row, vector<bool>(col, false));
        for(int i=0;i<row;++i){
            for(int j=0;j<col;++j){
                result = max(result, _explore_island(flags, grid, i, j));
            }
        }
        return result;
    }

    // problem 695
    int _explore_island(vector<vector<bool>> &flag, vector<vector<int>> &grid, int i, int j){
        if(i>=0 && i<grid.size() && j>=0 && j<grid[0].size() && !flag[i][j] && grid[i][j]){
            flag[i][j] = true;
            return 1+_explore_island(flag, grid, i-1,j)+_explore_island(flag, grid, i+1,j)+_explore_island(flag, grid, i, j-1)+_explore_island(flag, grid, i,j+1);
        }else{
            return 0;
        }
    }

    // problem 789
    bool escapeGhosts(vector<vector<int>>& ghosts, vector<int>& target) {
        int distance = abs(target[0]) + abs(target[1]);
        for(vector<int> &g:ghosts){
            if(distance >= abs(g[0]-target[0])+abs(g[1]-target[1])){
                return false;
            }
        }
        return true;
    }

    // problem 451
    struct lenCmp{
        bool operator()(string i, string j){
            return i.length() > j.length();
        }
    };

    // problem 451
    string frequencySort(string s) {
        map<char, int> count;
        for(char c:s){
            count[c]++;
        }
        vector<string> re_s;
        for(auto p:count){
            re_s.push_back(string(p.second, p.first));
        }
        struct lenCmp lenCmp1;
        sort(re_s.begin(), re_s.end(), lenCmp1);
        string result;
        for(string str:re_s){
            result+=str;
        }
        return result;
    }

    // problem 609
    vector<vector<string>> findDuplicate(vector<string>& paths) {
        int start, end, mark;
        map<string, vector<string>> record;
        for(string p: paths){
            mark = p.find(' ');
            string dir = p.substr(0, mark)+"/";
            for(int i = mark+1;i<p.length();++i) {
                if(p[i]==' ')
                    mark = i;
                else if(p[i]=='(')
                    start = i;
                else if(p[i] == ')') {
                    end = i;
                    record[p.substr(start + 1, end - start - 1)].push_back(dir+p.substr(mark+1, start-mark-1));
                }
            }
        }
        vector<vector<string>> result;
        for(const auto &p: record){
            if(p.second.size()!=1)
                result.push_back(p.second);
        }
        return result;
    }

    // problem 526
    void justTry(vector<bool> &flags, int N, int pos, int *count) {
        if(pos > N) {
            (*count)++;
            return;
        }
        for(int i=1;i<=N;++i){
            if(!flags[i] && (i%pos==0||pos%i==0)){
                flags[i] = true;
                justTry(flags,N, pos+1, count);
                flags[i] = false;
            }
        }
    }

    // problem 526
    int countArrangement(int N) {
        if(N == 0) return 0;
        int * count = new int(0), pos = 1;
        vector<bool> flags(N+1, false);

        justTry(flags, N, pos, count);
        return *count;
    }

    // problem 508
    vector<int> findFrequentTreeSum(TreeNode* root) {
        map<int, int> sums;

        computeSum(root, sums);

        vector<int> result;
        int max_s = INT32_MIN;
        for(auto &p : sums){
            if(max_s < p.second)
                max_s = p.second;
        }
        for(auto &p : sums){
            if(max_s == p.second)
                result.push_back(p.first);
        }
        return result;
    }

    // problem 508
    int computeSum(TreeNode *root, map<int, int> &sums){
        int result = 0;
        if(root != nullptr){
            result = root->val + computeSum(root->left, sums) + computeSum(root->right, sums);
            sums[result]++;
        }
        return result;
    }

    // problem 712
    int minimumDeleteSum(string s1, string s2) {
        int l_s1 = s1.length(), l_s2 = s2.length();
        vector<vector<int>> dp(l_s1+1, vector<int>(l_s2+1));
        for(int j=1;j<=l_s2;++j)
            dp[0][j] = dp[0][j-1] + s2[j-1];
        for(int i=1;i<=l_s1;++i){
            dp[i][0] = dp[i-1][0] + s1[i-1];
            for(int j = 1;j<=l_s2;++j){
                if(s1[i-1] == s2[j-1]){
                    dp[i][j] = dp[i-1][j-1];
                }else{
                    dp[i][j] = min(dp[i-1][j] + s1[i-1], dp[i][j-1] + s2[j-1]);
                }
            }
        }
        return dp[l_s1][l_s2];
    }

    // problem 865
    pair<int, TreeNode*> deep(TreeNode *pNode) {
        if(pNode== nullptr)
            return {0, nullptr};
        pair<int, TreeNode*> l = deep(pNode->left), r = deep(pNode->right);
        int d1 = l.first, d2 = r.first;
        return {max(d1, d2)+1, d1==d2?pNode:d1>d2?l.second:r.second};
    }

    // problem 865
    TreeNode* subtreeWithAllDeepest(TreeNode* root) {
        return deep(root).second;
    }

    // problem 389, char is same as number
    char findTheDifference(string s, string t) {
        char x = 0;
        for(char &c:s){
            x ^= c;
        }
        for(char &c:t){
            x ^= c;
        }
        return x;
    }

    // problem 462
    int minMoves2(vector<int>& nums) {
        nth_element(nums.begin(), nums.begin()+nums.size()/2, nums.end());
        int result = 0,mid = nums[nums.size()/2];
        for(int i:nums){
            result += abs(nums[i]-mid);
        }
        return result;
    }

    // problem 696
    int countBinarySubstrings(string s) {
        int result = 0;
        int pre=0,count = 1;
        for(int i=1;i<s.length();++i){
            if(s[i] == s[i-1]){
                count++;
            }else{
                result += min(count, pre);
                pre = count;
                count = 1;
            }
        }
        return result + min(count, pre);
    }

    // problem 653
    bool findTargetDFS(set<int> &s, TreeNode *root, int k){
        if(!root)
            return false;
        if(s.count(k-root->val)) return true;

        s.insert(root->val);
        return findTargetDFS(s, root->left, k) || findTargetDFS(s, root->right, k);
    }
    // problem 653
    bool findTarget(TreeNode* root, int k) {
        set<int> s;
        return findTargetDFS(s, root, k);
    }

    // problem 547
    void searchFriendCircle(vector<bool> &flags,vector<vector<int>> &M,int cur){
        flags[cur] = true;
        for(int i=0;i<flags.size();++i){
            if(!flags[i] && M[cur][i]){
                searchFriendCircle(flags, M, i);
            }
        }
    }
    // problem 547
    int findCircleNum(vector<vector<int>>& M) {
        int n = M.size(), count = 0, cur = 0;
        vector<bool> flags(n, false);
        for(int i = 0;i<n;++i) {
            for(int j = 0;j<n;++j) {
                if (!flags[i] && M[i][j]) {
                    count++;
                    searchFriendCircle(flags, M, i);
                }
            }
        }
        return count;
    }

    // problem 565
    int arrayNesting(vector<int>& nums) {
        int n = nums.size(), result = 0;
        vector<bool> flags(n, false);
        int cur = 0, count= 0;
        for(int i=0;i<n;++i){
            cur = i;
            count = 0;
            while(!flags[cur]){
                count++;
                flags[cur] = true;
                cur = nums[cur];
            }
            if(result < count)
                result = count;
        }
        return result;
    }

    // problem 860
    bool lemonadeChange(vector<int>& bills) {
        map<int, int> what_i_got;
        int change = 0;
        for(int i:bills){
            change = i - 5;
            if(change == 15){
                if(what_i_got[10]>0 && what_i_got[5]>0) {
                    what_i_got[10]--;
                    what_i_got[5]--;
                    change = 0;
                }else if(what_i_got[5]>2){
                    what_i_got[5] -= 3;
                    change = 0;
                }
            }else if(change == 10){
                if(what_i_got[10]>0) {
                    what_i_got[10]--;
                    change = 0;
                }else if(what_i_got[5] > 1){
                    what_i_got[5]-=2;
                    change = 0;
                }
            }else if(change == 5 && what_i_got[5]>0) {
                what_i_got[5]--;
                change = 0;
            }
            if(change==0)
                what_i_got[i]++;
            else
                return false;
        }
        return true;
    }

    // problem 421 TODO: Hard to understand the best resolution.
    int findMaximumXOR(vector<int>& nums) {
        int max = 0, n = nums.size(), tmp = 0;
        if(n > 999)
            return  2147483644;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                tmp = nums[i] ^ nums[j];
                if (max < tmp)
                    max = tmp;
            }
        }
        return max;
    }

    // problem 648
    string replaceWords(vector<string>& dict, string sentence) {
        vector<string> data;
        int n = sentence.length();
        int start = 0;
        for(int i=0;i<n;++i){
            if(sentence[i]==' '){
                data.push_back(sentence.substr(start, i-start));
                start = i+1;
            }
        }
        data.push_back(sentence.substr(start, n-start));

        string result = "";
        for(string &w:data){
            for(string &d:dict){
                int i = 0, len = d.size();
                for(;i<len;++i){
                    if(w[i]!=d[i]){
                        break;
                    }
                }
                if(i == len){
                    w = d;
                }
            }
            result += w + " ";
        }
        result = result.substr(0, result.size()-1);
        return result;
    }

    // problem 769
    int maxChunksToSorted(vector<int>& arr) {
        if(arr.empty())
            return 0;
        int count = 0, max_n = 0;
        for(int i=0;i<arr.size();++i){
            max_n = max(max_n, arr[i]);
            if(max_n == i)
                count++;
        }
        return count;
    }

    // problem 216
    void combination(vector<int> &com, vector<vector<int>> &result, int k, int n, int start) {
        if(!k && !n){
            result.push_back(com);
        }else{
            for(int i=start;i<10;++i){
                if(i<=n){
                    com.push_back(i);
                    combination(com, result, k-1, n-i, i+1);
                    com.pop_back();
                }
            }
        }
    }
    // problem 216
    vector<vector<int>> combinationSum3(int k, int n) {
        vector<vector<int>> result;
        vector<int> com;
        combination(com, result, k, n, 1);
        return result;
    }

    // problem 538
    void travel(TreeNode *root, int * sum){
        if(!root)
            return;
        if(root->right) travel(root->right, sum);
        root->val = (*sum += root->val);
        if(root->left) travel(root->left, sum);
    }

    // problem 538
    TreeNode* convertBST(TreeNode* root) {
        int * sum  = new int(0);
        travel(root, sum);
        return root;
    }

    // problem 796
    bool rotateString(string A, string B) {
        if(A.empty() && B.empty())
            return true;
        if(A.length() != B.length())
            return false;
        int n = A.length(), count = 0;
        for(int i=0;i<n;++i){
            if(A[i] == B[count]){
                count++;
            }else if(count){
                i--;
                count = 0;
            }
        }
        if(count == 0)
            return false;

        for(int i = 0;i<n-count;++i){
            if(A[i] == B[count]){
                count++;
            }else{
                return false;
            }
        }
        return true;
    }

    // problem 503
    vector<int> nextGreaterElements(vector<int>& nums) {
        int n = nums.size(), count = 0, max = INT32_MIN, index = 0, tmp;
        for(int i = 0;i<n;++i){
            if(max < nums[i]){
                index = i;
                max = nums[i];
            }
        }

        stack<int> stk;
        vector<int> result(n, -1);
        for(int i = index;(i+n)>=index+1;--i){
            tmp = (i+n)%n;
            while(!stk.empty() && stk.top() <= nums[tmp]){
                stk.pop();
            }
            if(!stk.empty()){
                result[tmp] = stk.top();
            }
            stk.push(nums[tmp]);
        }
        return result;
    }

    // problem 783
    void _minDiffBST(TreeNode* root,int *m, stack<int> &s){
        if(!root)
            return;
        if(root->right){
            _minDiffBST(root->right, m, s);
        }
        if(!s.empty() && *m > s.top()-root->val){
            *m = s.top() - root->val;
        }
        s.push(root->val);
        if(root->left)
            _minDiffBST(root->left,m,s);
    }

    // problem 783
    int minDiffInBST(TreeNode* root) {
        int *result = new int(INT32_MAX);
        stack<int> bigger;
        _minDiffBST(root, result, bigger);
        return *result;
    }

    // problem 530
    int getMinimumDifference(TreeNode* root) {
        int *result = new int(INT32_MAX);
        stack<int> bigger;
        _minDiffBST(root, result, bigger);
        return *result;
    }

    // problem 383
    bool canConstruct(string ransomNote, string magazine) {
        int dict[26]={0};
        for(char &c:magazine){
            dict[c-'a']++;
        }
        for(char &c:ransomNote){
            dict[c-'a']--;
        }
        for(auto &p:dict){
            if(p < 0)
                return false;
        }
        return true;
    }

    // problem 454
    int fourSumCount(vector<int>& A, vector<int>& B, vector<int>& C, vector<int>& D) {
        int count = 0, sum = 0, n = A.size();
        unordered_map<int, int> record;
        for(int i=0; i<n;++i){
            for(int j=0;j<n;++j){
                sum = A[i]+B[j];
                record[sum]++;
            }
        }

        for(int i=0; i<n;++i){
            for(int j=0;j<n;++j){
                sum = C[i]+D[j];
                count += record[-sum];
            }
        }
        return count;
    }

    // problem 733
    void replaceColor(vector<vector<int>> &image, int sr, int sc, int oldColor, int newColor) {
        if(sr>=0 && sc>=0 && sr<image.size() && sc<image[0].size() && image[sr][sc]==oldColor){
            image[sr][sc] = newColor;
            replaceColor(image, sr-1, sc, oldColor, newColor);
            replaceColor(image, sr+1, sc, oldColor, newColor);
            replaceColor(image, sr, sc-1, oldColor, newColor);
            replaceColor(image, sr, sc+1, oldColor, newColor);
        }
    }

    // problem 733
    vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int newColor) {
        int oldColor = image[sr][sc];
        if(newColor != oldColor)
            replaceColor(image, sr, sc, oldColor, newColor);
        return image;
    }

    // 78
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> result;
        int n;
        result.emplace_back();
        for(int &each:nums){
            n = result.size();
            for(int i=0;i<n;++i){
                vector<int> newSub = result[i];
                newSub.push_back(each);
                result.push_back(newSub);
            }
        }
        return result;
    }

    // problem 404
    int _sumOfLeftLeaves(TreeNode *parent, TreeNode *child) {
        if(child) {
            if(!(child->left||child->right) && parent->left == child){
                return child->val;
            }else{
                return _sumOfLeftLeaves(child, child->left)+_sumOfLeftLeaves(child, child->right);
            }
        }else
            return 0;
    }

    // problem 404
    int sumOfLeftLeaves(TreeNode* root) {
        if(root)
            return _sumOfLeftLeaves(root, root->left) + _sumOfLeftLeaves(root, root->right);
        else
            return 0;
    }

    // problem 869
    long countDigit(int N){
        long result = 0;
        for(;N>0;N/=10){
            result += pow(10, N%10);
        }
        return result;
    }

    // problem 869
    bool reorderedPowerOf2(int N) {
        long c = countDigit(N);
        for(int i = 0;i<32;++i){
            if(c == countDigit(1<<i)) return true;
        }
        return false;
    }

    // problem 445
    ListNode* reverseList(ListNode *pNode) {
        if(!pNode || !pNode->next){
            return pNode;
        }
        ListNode *pre = pNode;
        ListNode *cur = pNode->next;
        pre->next = nullptr;
        ListNode *next;
        while(cur){
            next = cur->next;
            cur->next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }

    // problem 445
    ListNode *_addTwoNumbers(ListNode *l1, ListNode *l2) {
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

    // problem 445
    ListNode* addTwoNumbersII(ListNode* l1, ListNode* l2) {
        ListNode *p1 = l1, *p2 = l2;
        int count = 0;
        l1 = reverseList(l1);
        l2 = reverseList(l2);
        return reverseList(_addTwoNumbers(l1, l2));
    }

    // problem 773
    int _computeId(vector<vector<int>> &board){
        int id = 0, digit = 100000;
        for(int i = 0;i<6;i++){
            id += board[i/3][i%3] * digit;
            digit /= 10;
        }
        return id;
    }

    // problem 773
    int slidingPuzzle(vector<vector<int>>& board) {
        int id = _computeId(board);
        unordered_set<string> records;
        queue<string> q;
        string id_s = to_string(id);
        if(id < 100000){
            id_s = "0"+id_s;
        }
        records.insert(id_s);
        q.push(id_s);
        int result = 0, index;
        int dire[4] = {1,-1,3,-3};
        string str, str_cp, target="123450";
        while(!q.empty()){
            for(int sz = q.size(); sz > 0;--sz){
                str = q.front();
                q.pop();
                if(str == target) return result;
                index = str.find('0');
                for(int k=0;k<4;++k){
                    int j = index+dire[k];
                    if(j<0 || j>5 || index==2 && j==3 || index==3 && j==2 ) continue;

                    str_cp = str;
                    char tmp  = str_cp[index];
                    str_cp[index] = str_cp[j];
                    str_cp[j] = tmp;

                    if(!records.count(str_cp)) {
                        records.insert(str_cp);
                        q.push(str_cp);
                    }
                }
            }
            ++result;
        }
        return -1;
    }

    // problem 697
    int findShortestSubArray(vector<int>& nums) {
        int degree = 0, result = nums.size(), tmp;

        unordered_map<int, int> count;
        unordered_map<int, int>  start;
        for(int i=0;i<nums.size();++i){
            count[nums[i]]++;

            if(degree < count[nums[i]]){
                // update result
                degree = count[nums[i]];
                result = start.count(nums[i])? i - start[nums[i]] + 1:1;
            }else if (degree ==  count[nums[i]]){
                // choose smaller result
                tmp = start.count(nums[i])? i - start[nums[i]] + 1:1;
                result = min(result, tmp);
            }

            if(!start.count(nums[i]))
                start[nums[i]] = i;
        }
        return result;
    }

    // problem 455
    int findContentChildren(vector<int>& g, vector<int>& s) {
        int result = 0;

        sort(g.begin(),g.end());
        sort(s.begin(),s.end());

        int i = 0, j=0;
        while(i < g.size() && j < s.size()){
            if(g[i] <= s[j]){
                result++;
                i++;
            }
            j++;
        }
        return result;
    }

    vector<ListNode*> splitListToParts(ListNode* root, int k) {
        ListNode * pNode = root;
        int size = 0;
        while(pNode){
            size++;
            pNode = pNode->next;
        }
        int part_len = size/k, addition = size % k;
        vector<ListNode*> result;
        pNode  = root;

        size = 0;
        ListNode * start_p = root;
        int tmp;
        while(pNode){
            size++;
            tmp = result.size();
            tmp =tmp >=addition? addition * (1 + part_len) + (tmp + 1 - addition) * part_len : (tmp+1)*(part_len+1);
            if(size == tmp){
                result.push_back(start_p);
                start_p = pNode;
                pNode = pNode->next;
                start_p->next = nullptr;
                start_p = pNode;
            }else{
                pNode = pNode->next;
            }
        }
        k -= size;
        while(k>0){
            result.push_back(nullptr);
            k--;
        }
        return result;
    }

    // problem 378
    // modified quick sort
    int kthSmallest(vector<vector<int>>& matrix, int k) {
        int n = matrix.size(), low = matrix[0][0], high = matrix[n-1][n-1], mid = 0;

        while(low < high) {
            mid = low + (high - low) / 2;
            int count = 0, j = n - 1;
            for (int i = 0; i < n; ++i) {
                while (j >= 0 && matrix[i][j] > mid) j--;
                count += (j + 1);
            }

            if (count < k) low = mid + 1;
            else high = mid;
        }
        return low;
    }

    // problem 539
    int _computeDiff(string a, string b){
        int ah = stoi(a.substr(0, 2));
        int bh = stoi(b.substr(0, 2));
        int am = stoi(a.substr(3, 2));
        int bm = stoi(b.substr(3, 2));
        return am - bm + 60*(ah-bh);
    }

    // problem 539
    int findMinDifference(vector<string>& timePoints) {
        int n = timePoints.size();
        sort(timePoints.begin(), timePoints.end());
        int mindiff = INT32_MAX;
        for(int i=0;i<n;i++){
            int diff = abs(_computeDiff(timePoints[(i-1+n)%n], timePoints[i]));
            diff = min(diff, 1440-diff);
            mindiff = min(mindiff, diff);
        }
        return mindiff;
    }

    // problem 684
    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
        vector<unordered_set<int>> record;
        vector<int> result;
        for(auto &edge:edges){
            bool no_set = true, combined = false;
            vector<unordered_set<int>>::iterator one = record.end();
            for(auto vertices = record.begin(); vertices!=record.end();vertices++) {
                if(vertices->count(edge[0])&&vertices->count(edge[1])){
                    no_set = false;
                    result = edge;
                    break;
                }
                else if(vertices->count(edge[0])){
                    vertices->insert(edge[1]);
                    no_set = false;
                    if(one!= record.end()){
                        vertices->insert(one->begin(),one->end());
                        combined = true;
                        break;
                    } else{
                        one = vertices;
                    }
                }else if(vertices->count(edge[1])){
                    vertices->insert(edge[0]);
                    no_set = false;
                    if(one!= record.end()){
                        vertices->insert(one->begin(),one->end());
                        combined = true;
                        break;
                    } else{
                        one = vertices;
                    }
                }
            }
            if(no_set){
                record.push_back(unordered_set<int>({edge[0], edge[1]}));
            }
            if(combined){
                record.erase(one);
            }
        }
        return result;
    }


    ListNode* head;
    // problem 382
    /** @param head The linked list's head.
    Note that the head is guaranteed to be not null, so it contains at least one node. */
    Solution(ListNode* head):head(head) {
    }

    /** Returns a random node's value. */
    int getRandom() {
        ListNode* p = head;
        int result = head->val;
        for(int i=1;p->next!=nullptr;++i){
            p = p->next;
            if(rand()%(i+1) == i) result = p->val;
        }
        return result;
    }

    // problem 477
    int totalHammingDistance(vector<int>& nums) {
        int count = 0, one = 0, size = nums.size();
        for(int i=0;i<32;++i){
            one = 0;
            for(int &n:nums){
                if(((n >> i)&1) == 1) one++;
            }
            count += one *(size-one);
        }
        return count;
    }

    // problem 167
    vector<int> twoSum(vector<int>& numbers, int target) {
        int i=0, j = numbers.size()-1;
        while(i<j){
            if(numbers[i]+numbers[j]>target) j--;
            if(numbers[i]+numbers[j]<target) i++;
            if(numbers[i]+numbers[j] == target) break;
        }
        return vector<int>({i+1,j+1});
    }

    // problem 646
    struct cmp646{
        bool operator()(vector<int> i, vector<int> j) {
            return i[1] < j[1];
        }
    };

    // problem 646
    int findLongestChain(vector<vector<int>>& pairs) {
        struct cmp646 cmp;
        sort(pairs.begin(), pairs.end(), cmp);
        int result = 0, cur_tail = INT32_MIN;
        for(auto &p :pairs){
            if(p[0]>cur_tail){
                result++;
                cur_tail = p[1];
            }
        }
        return result;
    }

    // problem 55
    bool canJump(vector<int>& nums) {
        if(nums.size()==0)
            return false;
        int max_range = nums[0], n = nums.size();
        for(int i = 0;i<=max_range;++i){
            if(max_range < i + nums[i]) max_range = i+nums[i];
            if(max_range >= n-1) return true;
        }
        return false;
    }

    // problem 554
    int leastBricks(vector<vector<int>>& wall) {
        if(wall.empty())
            return 0;
        int width = accumulate(wall[0].begin(), wall[0].end(), 0), height = wall.size(), least = height;
        vector<vector<int>> dp(height, vector<int>());
        unordered_map<int, int> count;
        for(int i=0;i<height;++i){
            for(int j=0;j<wall[i].size() - 1; ++j) {
                if (j) {
                    dp[i].push_back(dp[i][j - 1] + wall[i][j]);
                } else
                    dp[i].push_back(wall[i][j]);
                count[dp[i].back()]++;
            }
        }
        for(auto &p:count){
            if(height - p.second < least)
                least = height - p.second;
        }

        return least;
    }

    // problem 599
    vector<string> findRestaurant(vector<string>& list1, vector<string>& list2) {
        vector<string> result;

        map<string, int> record;
        int minSum = INT32_MAX;
        for(int i=0;i<list1.size();++i) record[list1[i]] = i;
        for(int i=0;i<list2.size();++i){
            if(record.count(list2[i]) && record[list2[i]] + i <= minSum){
                int tmp = record[list2[i]] + i;
                if(tmp < minSum){
                    minSum = tmp;
                    result.clear();
                }
                result.push_back(list2[i]);
            }
        }

        return result;
    }

    // problem 623 pending
    TreeNode* addOneRow(TreeNode* root, int v, int d)
    {
        if (d < 1)
            return nullptr;
        if (d == 1) {
            TreeNode *new_root =  new TreeNode(v);
            new_root->left = root;
            return new_root;
        }
        vector<TreeNode *> cur_level = {move(root)}, nex_level;
        while(--d > 1 & !cur_level.empty()){
            nex_level.clear();
            for (auto &node: cur_level) {
                if (node->left)
                    nex_level.push_back(move(node->left));
                if (node->right)
                    nex_level.push_back(move(node->right));
            }
            cur_level = nex_level;
        }
        for (auto node: cur_level) {
            TreeNode *new_left = new TreeNode(v);
            new_left ->left = node->left;
            node->left = new_left;

            TreeNode *new_right = new TreeNode(v);
            new_right->right = node->right;
            node->right = new_right;
        }
        return root;
    }

    // problem 830
    vector<vector<int>> largeGroupPositions(string S)
    {
        vector<vector<int>> result;
        int start = 0;
        for (int i = 1;i < S.length(); ++i) {
            if (S[i] != S[i-1]){
                if (i - start >= 3){
                    result.push_back({start, i - 1});
                }
                start = i;
            }
        }
        int size = S.length();
        if (size - start >= 3)
            result.push_back({start, size - 1});

        return result;
    }
};

#endif //LEETCODE_SOLUTION_H
