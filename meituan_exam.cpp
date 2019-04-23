//
// Created by zeal4u on 2019/4/23.
//
#include <iostream>
#include <vector>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include <list>

using namespace std;

// 我们称一个矩阵为黑白矩阵，当且仅当对于该矩阵中每一个位置如(i,j),其上下左右四个方向的数字相等，
// 即(i-1,j),(i+1,j),(i,j+1),(i,j-1)四个位置上的数字两两相等且均不等于(i,j)位置上的数字。(超出边界的格子忽略)
// 现在给出一个n*m的矩阵，我们想通过修改其中的某些数字，使得该矩阵成为黑白矩阵，问最少修改多少个数字。
void ModifyNum() {
  int n = 0, m = 0, val = 0;
  cin >> n >> m;
  unordered_map<int, int> record_even, record_odd;
  int max_even_count[2] = {-1, -1}, max_odd_count[2] = {-1, -1};
  int side_even_count[2] = {-1, -1}, side_odd_count[2] = {-1, -1};
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      cin >> val;
      if (((i + j) & 1) == 1) {
        record_odd[val]++;
        if (record_odd[val] > max_odd_count[0]) {
          max_odd_count[0] = record_odd[val];
          max_odd_count[1] = val;
        } else if (record_odd[val] > side_odd_count[0]) {
          side_odd_count[0] = record_odd[val];
          side_odd_count[1] = val;
        }
      } else {
        record_even[val]++;
        if (record_even[val] > max_even_count[0]) {
          max_even_count[0] = record_even[val];
          max_even_count[1] = val;
        } else if (record_even[val] > side_even_count[0]) {
          side_even_count[0] = record_even[val];
          side_even_count[1] = val;
        }
      }
    }
  }
  int res = 0;
  if (max_even_count[1] != max_odd_count[1])
    res = n * m - max_even_count[0] - max_odd_count[0];
  else {
    res = n * m - max({max_even_count[0] + side_odd_count[0], max_odd_count[0] + side_even_count[0]});
  }
  cout << res;
}

//
//给你一棵含有n个结点的树,编号为0…n-1，这n个结点都被染成了黑色或白色，显然，对于一棵树而言，我们每去掉一条边，就能把树分成两部分。
// 现在要求你把这棵树切开，使得每一个连通块内只有一个白色结点，问共有多少种切开的方式满足以上条件，如果被删除的边集不同，我们则认为两种方式不同，反之认为相同。
//
//请输出方案数对1000000007取模的结果

void WhiteBlackTree() {
  int n = 0;
  cin >> n;
  vector<list<int>> tree(n);
  int tmp = 0;
  for (int i = 1; i < n; ++i) {
    cin >> tmp;
    tree[tmp].push_back(i);
  }
  vector<int> is_white(n);
  for (int i = 0; i < n; ++i) {
    cin >> tmp;
    is_white[i] = tmp ^ 1;
  }
  int count = 0;
  for (int i = 0; i < n; ++i) {
    if (!is_white[i]) {
      for (int child : tree[i]) {
        count += is_white[child];
      }
    }
  }
  cout << count;
}


