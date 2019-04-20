//
// Created by zeal4u on 2019/3/4.
//

#include <stdexcept>
#include <vector>
#include <iostream>
#include "TwoQueueStack.h"

template <class T>
void TwoQueueStack<T>::push(T obj)
{
    q_[last_push_que_].push_back(obj);
}

template <class T>
T TwoQueueStack<T>::front()
{
    if (q_[last_push_que_].size() == 0)
        throw std::logic_error("The stack is empty!");
    while (q_[last_push_que_].size() != 1) {
        q_[last_push_que_ ^ 1].push_back(q_[last_push_que_].front());
        q_[last_push_que_].pop_front();
    }
    return q_[last_push_que_].front();
}

template <class T>
void TwoQueueStack<T>::pop()
{
    if (q_[last_push_que_].size() == 0)
        throw std::logic_error("The stack is empty!");
    else if (q_[last_push_que_].size() == 1)
        q_[last_push_que_].pop_front();
    else {
        front();
        q_[last_push_que_].pop_front();
    }
    last_push_que_ ^= 1;
}

template<>
void TwoQueueStack<int>::test_queue() {
    TwoQueueStack<int> s;
    std::vector<int> input({1,2,3,4,5});
    for (int i = 0; i < input.size() ; ++i) {
        s.push(input[i]);
    }
    for (int i = 0; i < input.size() - 1; ++i) {
        std::cout<<s.front()<<" ";
        s.pop();
    }

    std::vector<int> input2({6,7,8});
    for (int i = 0; i < input2.size(); ++i) {
        s.push(input2[i]);
    }
    for (int i = 0; i < input2.size() + 1; ++i) {
        std::cout<<s.front()<<" ";
        s.pop();
    }
}

template<class T>
int TwoQueueStack<T>::size() {
    return q_[0].size() + q_[1].size();
}
