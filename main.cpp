#include <iostream>
#include "Solution.h"
#include "FreqStack.h"

int main() {
    Solution solution;

    vector<int> rec1 = {5, 15, 8, 18};
    vector<int> rec2 = {0, 3, 7, 9};
    cout<<solution.isRectangleOverlap(rec1, rec2);
    return 0;
}