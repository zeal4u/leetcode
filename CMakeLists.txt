cmake_minimum_required(VERSION 3.12)
project(leetcode)

set(CMAKE_CXX_STANDARD 14)

add_executable(leetcode main.cpp Solution.cpp Solution.h Employee.h Node.h TreeNode.h MapSum.h MagicDictionary.h FreqStack.h PLRU.cpp PLRU.h TwoQueueStack.cpp TwoQueueStack.h TreeNode.cpp safe_stack.cpp safe_stack.h)