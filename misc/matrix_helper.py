#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: matrix_helper.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Sat 01 Feb 2020 01:18:09 PM CST
# ************************************************************************/

import numpy as np
import torch


def mat_1d_combination(u_j, u_0):
    return torch.cat((u_j, u_0))

def square_mat_diag_combination(var_j, var_0):
    """
    input:
    two square matrix, pytorch tensor
       (n*n) A = [[1, 2],
                  [2, 1]]
       (m*m) B = [[1, 1]
                  [1, 1]]

    output:
    get a sequare matrix
       (m+n)*(m+n)  C = [[A, 0],
                         [0, B]]
    """
    
    var_j_shape = var_j.shape
    var_0_shape = var_0.shape
    

    # square matrix check
    assert(var_j_shape[0]==var_j_shape[1])
    assert(var_0_shape[0]==var_0_shape[1])
    assert(var_j.dtype == var_0.dtype)

    n = var_j_shape[0]
    m = var_0_shape[0]
    dtype = var_j.dtype

    zeros = torch.zeros((m, n), dtype=dtype)
    left = torch.cat((var_j, zeros))

    zeros = torch.zeros((n, m), dtype=dtype)
    right = torch.cat((zeros, var_0))

    return torch.cat((left, right), 1)


if __name__ == "__main__":
    '''test'''
    u_j = torch.from_numpy(np.array([1. ,1.]))
    u_0 = torch.from_numpy(np.array([0. ,0.]))

    print(u_j)
    print(u_0)
    print(mat_1d_combination(u_j, u_0))

    var_j = np.array([[1, 2],
                       [2, 1]], dtype=float)
    var_j = torch.from_numpy(var_j)

    var_0 = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]], dtype=float)       
    var_0 = torch.from_numpy(var_0)


    print(var_j)
    print(var_0)
    print(square_mat_diag_combination(var_j, var_0))

