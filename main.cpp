#include <iostream>
#include "Solution.h"

int star_compute(int a, int b){
    return static_cast<int>(pow(a, b - 1) + pow(b, a - 1));
}
int main() {
    cout<<star_compute(1, star_compute(2,star_compute(3,star_compute(4,5))));
    return 0;
}