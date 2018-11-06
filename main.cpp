#include <iostream>
#include "Solution.h"
#include "FreqStack.h"

int main() {
    Solution solution;

    vector<int> nums = {3,3,1,0,4};
    cout<<solution.canJump(nums);
    return 0;
}