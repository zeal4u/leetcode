#include <iostream>
#include "Solution.h"

int main() {
    Solution solution;

    vector<vector<int>> circles = {{1,0,0,1},{0,1,1,0},{0,1,1,1},{1,0,1,1}};
    cout<<solution.findCircleNum(circles);
    return 0;
}