#include <iostream>
#include "Solution.h"
#include "FreqStack.h"



int main() {
    Solution solution;
    auto res = solution.letterCombinations("2312");
    for (auto &s :res)
        cout<<s<<endl;
    return 0;
}