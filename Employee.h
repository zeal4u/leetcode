//
// Created by zeal4u on 2018/9/18.
//

#ifndef LEETCODE_EMPLOYEE_H
#define LEETCODE_EMPLOYEE_H

#include <vector>


using namespace std;

class Employee {
public:
    // It's the unique ID of each node.
    // unique id of this employee
    int id;
    // the importance value of this employee
    int importance;
    // the id of direct subordinates
    vector<int> subordinates;
};


#endif //LEETCODE_EMPLOYEE_H
