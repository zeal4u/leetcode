#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <memory>

//#include "Solution.h"
#include "FreqStack.h"
#include "TwoQueueStack.h"
#include "Solution.h"

int number_of_1(int n) {
  int count = 0;
  while (n) {
    count++;
    n &= (n - 1);
  }
  return count;
}

void exchage_odd_even(std::vector<int> &nums, bool selector(const int)) {
  int j = 0;
  for (int i = 0; i < nums.size(); ++i) {
    if (selector(nums[i]))
      std::swap(nums[j++], nums[i]);
  }
}

struct data_type {
  char c:1;
  short s:1;
  int i:1;
  long l:1;
  long long ll:1;
};

int minNumberInRotateArray(vector<int> rotateArray) {
  if (rotateArray.size() == 0)
    return 0;
  int small = 0, big = rotateArray.size() - 1, mid = 0;
  while (small < big) {
    if (rotateArray[small] < rotateArray[big]) {
      break;
    }
    mid = (small + big) / 2;
    if (rotateArray[small] > rotateArray[mid]) {
      big = mid;
    } else if (rotateArray[big] < rotateArray[mid]) {
      small = mid + 1;
    } else {
      small++;
    }
  }
  return rotateArray[small];
}

string PrintMinNumber(vector<int> numbers) {
  vector<string> num_strs;
  for_each(numbers.begin(), numbers.end(), [&](int i) { num_strs.push_back(to_string(i)); });
  sort(num_strs.begin(), num_strs.end(),
       [](const string &a, const string &b) {
         int i = 0;
         for (; i < a.size() && i < b.size(); ++i) {
           if (a[i] < b[i])
             return true;
           else if (a[i] > b[i])
             return false;
         }
         if (i < a.size()) {
           while (a[i - 1] == a[i])
             i++;
           return a[i - 1] > a[i];
         } else if (i < b.size()) {
           while (b[i - 1] == b[i])
             i++;
           return b[i - 1] < b[i];
         }
         return false;
       }
  );
  string res;
  for_each(num_strs.begin(), num_strs.end(), [&](const string &s) { res += s; });
  return res;
}

int GetUglyNumber_Solution(int index) {
  if (index <= 0) return 0;
  vector<int> ugly_nums(index, 0);
  ugly_nums[0] = 1;
  int index_of_ugly2 = 0;
  int index_of_ugly3 = 0;
  int index_of_ugly5 = 0;
  for (int i = 1; i < index; ++i) {
    while (ugly_nums[index_of_ugly2] * 2 <= ugly_nums[i - 1]) {
      index_of_ugly2++;
    }
    while (ugly_nums[index_of_ugly3] * 3 <= ugly_nums[i - 1]) {
      index_of_ugly3++;
    }
    while (ugly_nums[index_of_ugly5] * 5 <= ugly_nums[i - 1]) {
      index_of_ugly5++;
    }
    ugly_nums[i] = min({ugly_nums[index_of_ugly2] * 2,
                        ugly_nums[index_of_ugly3] * 3,
                        ugly_nums[index_of_ugly5] * 5});
  }
  return ugly_nums[index - 1];
}

int GetUglyNumber_Solution2(int index) {
  if (index < 7)return index;
  vector<int> res(index);
  res[0] = 1;
  int t2 = 0, t3 = 0, t5 = 0, i;
  for (i = 1; i < index; ++i) {
    res[i] = min(res[t2] * 2, min(res[t3] * 3, res[t5] * 5));
    if (res[i] == res[t2] * 2)t2++;
    if (res[i] == res[t3] * 3)t3++;
    if (res[i] == res[t5] * 5)t5++;
  }
  return res[index - 1];
}


int GetNumberOfK(vector<int> data, int k) {
  int l = 0, r = data.size() - 1;
  int l_hand = l, r_hand = r;
  while (l <= r) {
    int mid = (l + r) / 2;
    if (data[mid] < k) {
      l = mid + 1;
    } else if (data[mid] > k) {
      r = mid - 1;
    } else if (mid - 1 >= 0 && data[mid - 1] != k || mid == 0) {
      l_hand = mid;
      break;
    } else {
      r = mid - 1;
    }
  }
  if (data[l_hand] != k)
    return 0;
  l = l_hand, r = data.size() - 1;
  while (l <= r) {
    int mid = (l + r) / 2;
    if (data[mid] < k) {
      l = mid + 1;
    } else if (data[mid] > k) {
      r = mid - 1;
    } else if (mid + 1 < data.size() && data[mid + 1] != k || mid == data.size() - 1) {
      r_hand = mid;
      break;
    } else {
      l = mid + 1;
    }
  }

  if (data[r_hand] != k)
    return 0;
  return r_hand - l_hand + 1;
}

class MatrixSearch {
 private:
  int rows_;
  int cols_;
 public:
  bool valid(int index, bool *flags) {
    if (flags == nullptr)
      return false;
    return index >= 0 && index < rows_ * cols_ && !flags[index];
  }

  bool searchPath(char *matrix, bool *flags, int x, int y, char *str) {
    if (str == nullptr || matrix == nullptr || flags == nullptr)
      return false;
    if (*str == '\0')
      return true;
    vector<vector<int>> position = {{x,     y + 1},
                                    {x,     y - 1},
                                    {x - 1, y},
                                    {x + 1, y}};
    vector<int> next = {x * cols_ + y + 1, x * cols_ + y - 1, (x - 1) * cols_ + y, (x + 1) * cols_ + y};
    bool res = false;
    flags[x * cols_ + y] = true;
    for (int i = 0; i < 4; ++i) {
      if (valid(next[i], flags) && *str == matrix[next[i]]) {
        res |= searchPath(matrix, flags, position[i][0], position[i][1], str + 1);
      }
    }
    if (!res)
      flags[x * cols_ + y] = false;
    return res;
  }

  bool hasPath(char *matrix, int rows, int cols, char *str) {
    if (str == nullptr || matrix == nullptr || rows * cols == 0)
      return false;
    rows_ = rows, cols_ = cols;
    bool *flags = new bool[rows * cols];
    memset(flags, false, sizeof(bool) * rows * cols);
    bool res = false;
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        if (*str == matrix[i * cols + j])
          res |= searchPath(matrix, flags, i, j, str + 1);
      }
    }
    return res;
  }
};

class RobotSearch {
 private:
  int rows_;
  int cols_;
  int threshold_;
 public:
  int sumOfDigit(int i) {
    int sum = 0;
    while (i) {
      sum += i % 10;
      i /= 10;
    }
    return sum;
  }

  bool valid(int x, int y) {
    return x >= 0 && x < rows_ && y >= 0 && y < cols_ && (sumOfDigit(x) + sumOfDigit(y)) <= threshold_;
  }

  void search(bool *flags, int x, int y, int &count) {
    if (!flags[x * cols_ + y] && valid(x, y)) {
      flags[x * cols_ + y] = true;
      vector<vector<int>> position = {{x,     y - 1},
                                      {x,     y + 1},
                                      {x - 1, y},
                                      {x + 1, y}};
      count++;
      for (int i = 0; i < position.size(); ++i) {
        search(flags, position[i][0], position[i][1], count);
      }
    }
  }

  int movingCount(int threshold, int rows, int cols) {
    rows_ = rows, cols_ = cols, threshold_ = threshold;
    int count = 0;
    bool *flags = new bool[rows * cols];
    memset(flags, false, sizeof(bool) * rows * cols);
    search(flags, 0, 0, count);
    delete[] flags;
    return count;
  }
};

void permutate(set<string> &store, string str, int index) {
  store.insert(str);
  for (int i = index; i < str.size(); ++i) {
    swap(str[index], str[i]);
    permutate(store, str, index + 1);
    swap(str[index], str[i]);
  }
}

vector<string> Permutation(string str) {
  if (str.size() == 0)
    return {};
  set<string> res;
  permutate(res, str, 0);
  vector<string> true_res(res.begin(), res.end());
  return true_res;
}

bool IsSameTree(TreeNode *pRoot1, TreeNode *pRoot2) {
  if (pRoot1 == nullptr && pRoot2 == nullptr)
    return true;
  else if (pRoot1 == nullptr || pRoot2 == nullptr)
    return false;
  return pRoot1->val == pRoot2->val &&
         IsSameTree(pRoot1->left, pRoot2->left) &&
         IsSameTree(pRoot1->right, pRoot2->right);
}

bool HasSubtree(TreeNode *pRoot1, TreeNode *pRoot2) {
  if (pRoot1 == nullptr && pRoot2 == nullptr)
    return true;
  else if (pRoot1 == nullptr || pRoot2 == nullptr)
    return false;
  return IsSameTree(pRoot1, pRoot2) ||
         HasSubtree(pRoot1->left, pRoot2) ||
         HasSubtree(pRoot1->right, pRoot2);
}

bool isUnsigned(char *str, char *end) {
  if (str == nullptr || end == nullptr || str == end)
    return false;
  for (int i = 0; i + str != end; ++i) {
    if (*(i + str) < '0' || *(i + str) > '9')
      return false;
  }
  return true;
}

bool isSigned(char *str, char *end) {
  if (str == nullptr || end == nullptr || str == end)
    return false;
  if (*str == '+' || *str == '-')
    return isUnsigned(str + 1, end);
  else
    return isUnsigned(str, end);
}

bool isNumeric(char *str) {
  if (str == nullptr || *str == '\0')
    return false;
  int dot_pivot = -1, e_pivot = -1;
  for (int i = 0; *(i + str) != '\0'; ++i) {
    if (*(i + str) == '.') {
      if (dot_pivot == -1)
        dot_pivot = i;
      else
        return false;
    }
    if (*(i + str) == 'e' || *(i + str) == 'E') {
      if (e_pivot == -1)
        e_pivot = i;
      else
        return false;
    }
  }
  if (dot_pivot != -1 && e_pivot != -1 && e_pivot < dot_pivot)
    return false;
  bool res = true;
  if (dot_pivot != -1 || e_pivot != -1) {
    if (dot_pivot != -1) {
      res &= isSigned(str, str + dot_pivot);
      if (e_pivot == -1)
        res &= isUnsigned(str + dot_pivot + 1, str + strlen(str));
    }
    if (e_pivot != -1) {
      if (dot_pivot != -1)
        res &= isUnsigned(str + dot_pivot + 1, str + e_pivot);
      else
        res &= isSigned(str, str + e_pivot);
      res &= isSigned(str + e_pivot + 1, str + strlen(str));
    }
  } else {
    res &= isSigned(str, str + strlen(str));
  }
  return res;
}

class TreeSer {
 private:
  void Traversal(TreeNode *root, int **store) {
    if (root == nullptr) {
      **store = INT32_MIN;
      (*store)++;
      return;
    }
    **store = root->val;
    (*store)++;
    Traversal(root->left, store);
    Traversal(root->right, store);
  }

  TreeNode *Rebuild(int **store) {
    if (**store == INT32_MIN) {
      (*store)++;
      return nullptr;
    }
    TreeNode *root = new TreeNode(**store);
    (*store)++;
    root->left = Rebuild(store);
    root->right = Rebuild(store);
    return root;
  }

 public:
  const static int MAX_LEN = 4096;

  char *Serialize(TreeNode *root) {
    if (root == nullptr)
      return nullptr;
    int *store = new int[MAX_LEN];
    int *origin = store;
    Traversal(root, &store);
    return reinterpret_cast<char *>(origin);
  }

  TreeNode *Deserialize(char *str) {
    int *store = reinterpret_cast<int *>(str);
    return Rebuild(&store);
  }
};

void mergeSort(vector<int> &data, int l, int r, int &res) {
  if (r <= l)
    return;
  int mid = (l + r) / 2;
  mergeSort(data, l, mid, res);
  mergeSort(data, mid + 1, r, res);

  int i = l, j = mid + 1;
  vector<int> copy;
  while (i <= mid && j <= r) {
    if (data[i] <= data[j]) {
      copy.push_back(data[i++]);
    } else {
      res += mid + 1 - i;
      copy.push_back(data[j++]);
    }
  }
  if (i <= mid) {
    copy.insert(copy.end(), data.begin() + i, data.begin() + mid + 1);
  }
  if (j <= r) {
    copy.insert(copy.end(), data.begin() + j, data.begin() + r + 1);
  }
  for (int k = l; k <= r; ++k)
    data[k] = copy[k - l];
}

int InversePairs(vector<int> data) {
  int res = 0;
  mergeSort(data, 0, data.size() - 1, res);
  return res;
}

class InPairs {
 public:
  static constexpr int P = 1000000007;
  vector<int>::iterator it;

  int InversePairs(vector<int> data) {
    it = data.begin();
    if (data.empty())return 0;
    vector<int> dup(data);
    return merge_sort(data.begin(), data.end(), dup.begin());
  }
  //template<class RanIt>
  using RanIt = vector<int>::iterator;

  int merge_sort(const RanIt &begin1, const RanIt &end1, const RanIt &begin2) {
    int len = end1 - begin1;
    if (len < 2)return 0;
    int mid = (len + 1) >> 1;
    auto m1 = begin1 + mid, m2 = begin2 + mid;
    auto i = m1, j = end1, k = begin2 + len;
    int ans = (merge_sort(begin2, m2, begin1) + merge_sort(m2, k, m1)) % P;
    for (--i, --j, --k; i >= begin1 && j >= m1; --k) {
      if (*i > *j) {
        *k = *i, --i;
        (ans += j - m1 + 1) %= P;
      } else *k = *j, --j;
    }
    if (i >= begin1)copy(begin1, i + 1, begin2);
    else copy(m1, j + 1, begin2);
    return ans;
  }

};

int getSum(int a, int b) {
  int sum = 0, carry = 0;
  do {
    sum = a ^ b;
    carry = (a & b) << 1;
    a = sum;
    b = carry;
  } while (carry);
  return sum;
}

// This is the interface that allows for creating nested lists.
// You should not implement it, or speculate about its implementation
class NestedInteger {
 private:
  vector<NestedInteger> nested_;
  int unnested_ = 0;
 public:
  NestedInteger() {};
  NestedInteger(int i): unnested_(i){};
  NestedInteger(const vector<int> &v) {
    for (int i:v) {
      nested_.push_back(NestedInteger(i));
    }
  };

  void add(const NestedInteger &ni) {
    nested_.push_back(ni);
  }
  // Return true if this NestedInteger holds a single integer, rather than a nested list.
  bool isInteger() const {
    return nested_.size() == 0;
  }

  // Return the single integer that this NestedInteger holds, if it holds a single integer
  // The result is undefined if this NestedInteger holds a nested list
  int getInteger() const {
    return unnested_;
  }

  // Return the nested list that this NestedInteger holds, if it holds a nested list
  // The result is undefined if this NestedInteger holds a single integer
  vector<NestedInteger> &getList(){
    return nested_;
  }
};

class NestedIterator {
 private:
  struct IndexedNestedIntegerIterator {
    vector<NestedInteger>::iterator cur_it_;
    vector<NestedInteger>::iterator end_;
  };

  stack<IndexedNestedIntegerIterator> storage_;
  vector<NestedInteger>::iterator cur_it_;
  vector<NestedInteger>::iterator end_;
 public:
  NestedIterator(vector<NestedInteger> &nestedList) {
    cur_it_ = nestedList.begin();
    end_ = nestedList.end();
  }

  int next() {
    return (cur_it_++)->getInteger();
  }

  bool hasNext() {
    while (cur_it_ == end_) {
      if (storage_.size() > 0) {
        IndexedNestedIntegerIterator parent = storage_.top();
        cur_it_ = parent.cur_it_;
        end_ = parent.end_;
        if (cur_it_ != end_)
          cur_it_++;
        storage_.pop();
      } else
        return false;
    }
    while (!cur_it_->isInteger()) {
      IndexedNestedIntegerIterator inia;
      inia.cur_it_ = cur_it_;
      inia.end_ = end_;
      storage_.push(inia);
      // attention! do not update cur_it first!
      end_ = cur_it_->getList().end();
      cur_it_ = cur_it_->getList().begin();
    }
    return true;
  }
};

vector<vector<int>> generate(int numRows) {
  if (numRows <= 0)
    return {};
  vector<vector<int>> result;
  for (int i = 0; i < numRows; i++) {
    vector<int> row(i+1, 1);
    for (int j = 1; j < i; j++) {
      row[j] = result[i-1][j-1] + result[i-1][j];
    }
    result.push_back(move(row));
  }
  return result;
}

int problem1() {
  int n = 0, m = 0;
  std::cin >> n >> m;
  std::vector<int> nums(n, 0);
  for (int i = 0; i < n; ++i) {
    std::cin >> nums[i];
  }
  int t = 0, x = 0;
  for (int i = 0; i < m; ++i) {
    std::cin >> t >> x;
    if (t == 0) {
      sort(nums.begin(), nums.begin() + x, less<int>());
    } else {
      sort(nums.begin(), nums.begin() + x, greater<int>());
    }
  }
  for_each(nums.begin(), nums.end(), [](int x) {std::cout << x << " ";});
  std::cout << std::endl;
  return 0;
}

int max_circle_value() {
  int n = 0;
  std::cin >> n;
  std::vector<int> nums(n);
  for (int i = 0; i < n; ++i) {
    std::cin >> nums[i];
  }
  int sum = INT32_MAX;
  for (int i = 0; i < n; ++i) {
    int cur_sum = 0;
    for (int j = 0; j < n; ++j) {
      cur_sum += abs(nums[j] - ((i + j) % n) - 1);
      if (cur_sum >= sum)
        break;
    }
    if (cur_sum < sum)
      sum = cur_sum;
  }
  std::cout << sum << std::endl;
  return 0;
}