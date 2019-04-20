//
// Created by zeal4u on 2019/3/25.
//

#ifndef LEETCODE_SAFE_STACK_H
#define LEETCODE_SAFE_STACK_H


#include <memory>
#include <atomic>

template <class T>
class SafeStack{
 private:
  struct Node{
    std::shared_ptr<T> data_;
    std::shared_ptr<Node> next_;

    Node(const T& data): data_(std::make_shared<T>(data)) {}
  };

  std::shared_ptr<Node> head_;
 public:
  void push(const T& data) {
    std::shared_ptr<Node> new_node = std::make_shared<Node>(data);
    new_node->next = head_.get();
    while (!std::atomic_compare_exchange_weak(&head_, &new_node->next_, new_node));
  }

  std::shared_ptr<T> pop() {
    std::shared_ptr<Node> old_head = head_.get();
    while (old_head && !std::atomic_compare_exchange_weak(&head_, &old_head, old_head->next_));
    return old_head ? old_head->data_ : std::shared_ptr<T>();
  }
};

template <class T>
class LockFreeStack{
 private:
  struct Node;
  struct CountedNodePtr{
    int external_count_;
    Node* ptr;
  };

  struct Node{
    std::shared_ptr<T> data_;
    std::atomic<int> internal_count_;
    CountedNodePtr next_;

    Node(const T& data): data_(std::make_shared<T>(data)), internal_count_(0) {}
  };

  std::atomic<CountedNodePtr> head_;

  void increase_head_count(CountedNodePtr &old_counter) {
    CountedNodePtr new_counter;
    do {
      new_counter = old_counter;
      ++new_counter.external_count_;
    }
    while (!head_.compare_exchange_strong(old_counter, new_counter));
    old_counter.external_count_ = new_counter.external_count_;
  }

 public:
  ~LockFreeStack() {
    while(pop());
  }

  void push(const T& data) {
    CountedNodePtr new_node;
    new_node.ptr = new Node(data);
    new_node.external_count_ = 1;
    new_node.ptr->next_ = head_.load();
    while (!head_.compare_exchange_weak(new_node.ptr->next_, new_node));
  }

  std::shared_ptr<T> pop() {
    CountedNodePtr old_head = head_.load();
    while (true) {
      increase_head_count(old_head);
      Node *const ptr = old_head.ptr;
      if (!ptr) {
        return std::shared_ptr<T>();
      }
      if (head_.compare_exchange_strong(old_head, ptr->next_)) {
        std::shared_ptr<T> res;
        res.swap(ptr->data_);

        const int count_increase = old_head.external_count_ - 2;

        if (ptr->internal_count_.fetch_add(count_increase) == -count_increase) {
          delete ptr;
        }
        return res;
      } else if (ptr->internal_count_.fetch_sub(1) == 1) {
        delete ptr;
      }
    }
  }
};


#endif //LEETCODE_SAFE_STACK_H
