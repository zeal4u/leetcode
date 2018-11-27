#include <iostream>
#include "Solution.h"
#include "FreqStack.h"

int main() {
    Solution solution;

    vector<string> words = {"time", "me", "bell"};
    cout<<solution.minimumLengthEncoding(words);
    return 0;
}