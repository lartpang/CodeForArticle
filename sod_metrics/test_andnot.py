# -*- coding: utf-8 -*-
# @Time    : 2020/11/29
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
import time

import numpy as np


# 快速统计numpy数组的非零值建议使用np.count_nonzero，一个简单的小实验
def cal_andnot(size):
    a = np.random.randn(size, size)
    b = np.random.randn(size, size)
    a = a > 0
    b = b < 0
    start = time.time()
    a_and_b_mul = a * b
    _a_and__b_mul = (1 - a) * (1 - b)
    print(time.time() - start)
    start = time.time()
    a_and_b_and = a & b
    _a_and__b_and = ~a & ~b
    print(time.time() - start)


if __name__ == '__main__':
    cal_andnot(1000)
    # 0.0036919116973876953
    # 0.0005502700805664062
