#include <iostream>
#include "Solution.h"
#include "FreqStack.h"

int main() {
    Solution solution;

    //vector<pair<string, string>> equations = {{"x1","x2"},{"x2","x3"},{"x1","x4"},{"x2","x5"}};
    vector<pair<string, string>> equations = {{"a","b"},{"b","c"},{"b","d"}};
    vector<double> values = {3.0,0.5,3.4};
    //vector<pair<string, string>> queries = {{"x2","x4"},{"x1","x5"},{"x1","x3"},{"x5","x5"},{"x5","x1"},{"x3","x4"},{"x4","x3"},{"x6","x6"},{"x0","x0"}};
    vector<pair<string, string>> queries = {{"a", "d"}};
    solution.calcEquation(equations, values, queries);
    return 0;
}