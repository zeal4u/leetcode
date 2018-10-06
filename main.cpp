#include <iostream>
#include "Solution.h"

int main() {
    Solution solution;

    string S = "this apple is sweet";
    string T = "this apple is sour";
    vector<string> result = solution.uncommonFromSentences(S,T);
    for(string s: result){
        cout<< s<<" ";
    }
    return 0;
}