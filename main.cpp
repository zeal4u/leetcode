#include <iostream>
#include "Solution.h"
#include "FreqStack.h"

int main() {
    Solution solution;

    vector<vector<int>> edges = {{9,10},{5,8},{2,6},{1,5},{3,8},{4,9},{8,10},{4,10},{6,8},{7,9}};
    solution.findRedundantConnection(edges);
    return 0;
}