//
// Created by zeal4u on 2018/10/12.
//

#ifndef LEETCODE_FREQSTACK_H
#define LEETCODE_FREQSTACK_H

#include <unordered_map>
#include <stack>

using namespace std;

class FreqStack {
private:
    unordered_map<int, stack<int>> data;
    unordered_map<int, int> count;
    int max_freq;
public:
    FreqStack():data(), count(), max_freq(){

    }

    void push(int x) {
        count[x]++;
        data[count[x]].push(x);
        max_freq = max(max_freq, count[x]);
    }

    int pop() {
        int result = data[max_freq].top();
        data[count[result]--].pop();
        if(data[max_freq].empty()){
            max_freq--;
        }
        return result;
    }
};
#endif //LEETCODE_FREQSTACK_H
