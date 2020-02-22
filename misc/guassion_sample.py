#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: guassion_sample.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Sat 22 Feb 2020 07:22:40 PM CST
# ************************************************************************/

import numpy as np

mean = np.zeros(5, dtype=float)
conv = np.identity(5, dtype=float) * 0.1
size = 10

val = np.random.multivariate_normal(mean=mean, cov=conv, size=size)

print("mean", np.shape(mean))
print(mean)
print("conv", np.shape(conv))
print(conv)
print("val", np.shape(val))
print(val)
