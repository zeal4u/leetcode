//
// Created by zeal4u on 2019/3/4.
//

#ifndef LEETCODE_TWOQUEUESTACK_H
#define LEETCODE_TWOQUEUESTACK_H

#include <stack>

template <class T>
class TwoQueueStack {
public:
    TwoQueueStack() : last_push_que_(0) {}

    void push(T);
    T front();
    void pop();
    int size();
    static void test_queue();
private:
    std::deque<T> q_[2];
    int last_push_que_;
};



#endif //LEETCODE_TWOQUEUESTACK_H
