#include <iostream>
#include "Solution.h"
#include "FreqStack.h"

int main() {
    Solution solution;

    vector<pair<int, int>> pre = {{0, 1}, {0, 2}, {1, 2}};
    cout<<solution.canFinish(2, pre);
    return 0;
}