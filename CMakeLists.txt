cmake_minimum_required(VERSION 3.12)
project(leetcode)

set(CMAKE_CXX_STANDARD 14)

add_executable(leetcode interviewproblem.cpp Solution.cpp Solution.h Employee.h Node.h TreeNode.h MapSum.h MagicDictionary.h FreqStack.h TwoQueueStack.cpp TwoQueueStack.h TreeNode.cpp safe_stack.cpp safe_stack.h binary_tree_problems.h main.cpp)