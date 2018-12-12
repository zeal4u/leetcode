#include <iostream>
#include "Solution.h"
#include "FreqStack.h"

int main() {
    Solution solution;

    vector<int> nums =  {0, 5, 9, -8, -1};
    cout<<solution.findKthLargest(nums, 5);
    return 0;
}