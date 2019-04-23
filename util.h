//
// Created by zeal4u on 2019/4/22.
//

#ifndef LEETCODE_UTIL_H
#define LEETCODE_UTIL_H

#include <string>
#include <vector>

namespace tools {
void KMP(const std::string str, std::vector<int> &next);

int IsSubStr(const std::string &a, const std::string &b);
} // namespace tools

#endif //LEETCODE_UTIL_H
