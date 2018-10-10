//
// Created by zeal4u on 2018/10/10.
//

#ifndef LEETCODE_MAPSUM_H
#define LEETCODE_MAPSUM_H

#include <string>
#include <map>

using namespace std;
// problem 677
class MapSum {
private:
    map<string, int> data;

public:
    /** Initialize your data structure here. */
    MapSum():data() {
    }

    void insert(string key, int val) {
        data[key] = val;
    }

    int sum(string prefix) {
        int result = 0, n = prefix.length();
        for(auto &p:data){
            if(p.first.substr(0, n) == prefix){
                result += p.second;
            }
        }
        return result;
    }
};


#endif //LEETCODE_MAPSUM_H
