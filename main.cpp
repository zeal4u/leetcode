#include <iostream>
#include "Solution.h"
#include "FreqStack.h"

int main() {
    Solution solution;

    string s = "abababa";
    string p = "ab";

    solution.findAnagrams(s, p);

    return 0;
}