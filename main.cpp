#include <iostream>
#include "Solution.h"
#include "FreqStack.h"

int main() {
    Solution solution;

    vector<string> list1 = {"Shogun","Piatti","Burger King","KFC"};
    vector<string> list2 = {"Piatti","Shogun"};
    auto result = solution.findRestaurant(list1, list2);
    for(auto &r :result)
        cout<<r<<" ";
    return 0;
}