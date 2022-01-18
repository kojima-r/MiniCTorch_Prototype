#include <gtest/gtest.h>
//#include "sum.hpp"

int sum(int a, int b) {
    return a + b;
}

TEST(MyTestCase, TestSum) {
    EXPECT_EQ(3, sum(1, 2));
}
