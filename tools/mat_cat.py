#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: mat_cat.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Sat 01 Feb 2020 01:45:30 PM CST
# ************************************************************************/

import numpy as np

def class_mean(data, label):
    "u_j and var_j"

    assert np.shape(data)[0] == np.shape(label)[0]
    data_class = np.unique(label)

    contain = []
    for i in range(len(data_class)):
        contain.append([])
    
    for i in range(len(label)):
        class_index = label[i]
        contain[class_index].append(data[i])
    
    class_dataset = []
    for it in contain:
       class_dataset.append(np.array(it)) 

    class_mean = np.zeros((len(class_dataset), np.shape(data)[1]))

    for i in range(len(class_dataset)):
        it = class_dataset[i]
        class_mean[i] = it.mean(axis=0)

    return class_mean


 
def all_mean(data, label):
    '''compute the all mean and var'''
    all_mean = np.ones((len(np.unique(label)), np.shape(data)[1]), dtype=float)
    all_mean_1d = data.mean(axis=0)
    all_mean = all_mean*all_mean_1d

    return all_mean



if __name__ == "__main__":

    import torch
    data = [
            [1, 1, 1],
            [1, 1, 1],
            [0, 0, 0]
            ]
    label = [1, 0, 1]

    data = np.array(data)

    class_mean = class_mean(data, label)
    class_mean = torch.from_numpy(class_mean)
    print(class_mean)

    all_mean = all_mean(data, label)
    all_mean = torch.from_numpy(all_mean)
    print(all_mean)

    # split and cat
    c_dim = 2
    class_mean = class_mean[:, :c_dim]
    all_mean = all_mean[:, c_dim:]
    class_mean = torch.cat((class_mean, all_mean), 1)

    print(class_mean)

