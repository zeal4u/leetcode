#include <iostream>
#include "Solution.h"
#include "FreqStack.h"

int main() {
    Solution solution;

    vector<int> nums =  {0, 5, 9, -8, -1};
    cout<<solution.maximumProduct(nums)<<endl;
    return 0;
}