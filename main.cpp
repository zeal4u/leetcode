#include <iostream>
#include "Solution.h"
#include "FreqStack.h"

int main() {
    Solution solution;

    vector<int> nums = {1, 2, 3, 5};
    cout<<solution.canPartition(nums)<<endl;
    return 0;
}