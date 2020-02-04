#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: mean.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Wed 05 Feb 2020 03:24:32 PM CST
# ************************************************************************/

import torch
import numpy as np

def get_class_mean(data, label):
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


def get_all_mean(data, label):
    '''compute the all mean and var'''
    all_mean = np.ones((len(np.unique(label)), np.shape(data)[1]), dtype=float)
    all_mean_1d = data.mean(axis=0)
    all_mean = all_mean*all_mean_1d

    return all_mean


