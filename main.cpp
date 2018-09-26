#include <iostream>
#include "Solution.h"

int main() {
    Solution solution;

    vector<vector<int>> A = {{0,0,1,1},{1,0,1,0},{1,1,0,0}};
    cout<<solution.matrixScore(A);
    return 0;
}