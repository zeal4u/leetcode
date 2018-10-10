//
// Created by zeal4u on 2018/10/10.
//

#ifndef LEETCODE_MAGICDICTIONARY_H
#define LEETCODE_MAGICDICTIONARY_H

#include <string>
#include <vector>
#include <map>

using namespace std;

// problem 676
class MagicDictionary {
private:
    map<int, vector<string>> data;
public:
    /** Initialize your data structure here. */
    MagicDictionary():data() {

    }

    /** Build a dictionary through a list of words */
    void buildDict(vector<string> dict) {
        for(string &s:dict){
            data[s.length()].push_back(s);
        }
    }

    /** Returns if there is any word in the trie that equals to the given word after modifying exactly one character */
    bool search(string word) {
        int len = word.length(), count = 0;
        vector<string> &list = data[len];
        for(string &s:list){
            count = 0;
            for(int i=0;i<len;++i){
                if(s[i]!=word[i])
                    count++;
                if(count == 2){
                   break;
                }
            }
            if(count==1)
                return true;
        }
        return false;
    }
};


#endif //LEETCODE_MAGICDICTIONARY_H
