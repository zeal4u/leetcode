#include <iostream>
#include "Solution.h"
#include "FreqStack.h"

int main() {
    Solution solution;

    vector<int> candidates={2,3,7};
    auto result = solution.combinationSum(candidates, 18);
    for (auto &row:result){
        for (auto &i:row)
            cout<< i << " ";

        cout<<endl;
    }
    return 0;
}