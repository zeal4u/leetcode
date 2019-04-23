//
// Created by zeal4u on 2019/4/22.
//
#include "util.h"

namespace tools {
void KMP(const std::string str, std::vector<int> &next) {
  if (str.size() == 0)
    return;
  if (next.size() != str.size()) {
    next.resize(str.size());
  }
  next[0] = -1;
  int k = -1, j = 0;
  while (j < str.size()) {
    if (k == -1 || str[j] == str[k]) {
      k++, j++;
      next[j] = k;
    } else {
      k = next[k];
    }
  }
}

int IsSubStr(const std::string &a, const std::string &b) {
  if (a.size() < b.size())
    return -1;
  std::vector<int> next(b.size());
  KMP(b, next);

  int k = 0, j = 0;
  while (j < a.size()) {
    if (k == -1 || a[j] == b[k]) {
      j++,k++;
      if (k == b.size())
        return j - k;
    } else {
      k = next[k];
    }
  }
  return -1;
}
} // namespace tools

