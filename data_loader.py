#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: data_loader.py
#	> Author: Yang Zhang
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Mon 20 Jan 2020 06:18:35 PM CST
# ************************************************************************/


import torch
import torch.utils.data as data
import os
import numpy as np
import copy
import random


class feats_data_loader(data.Dataset):
    def __init__(self, npz_path="data/feats.npz", dataset_name="vox"):
        assert os.path.exists(npz_path) == True
        print("hi")

        self.dataset_name = dataset_name

        _data = np.load(npz_path, allow_pickle=True)['feats']
        _label = np.load(npz_path, allow_pickle=True)['spkers']

        self._data = _data
        self._label = _label

        # class
        self.data_class = np.unique(_label)

        # class counter: the number of each class
        self.class_counter = np.zeros(len(self.data_class), dtype=int)
        for i in self._label:
            self.class_counter[i] += 1

        print("dataset: {}".format(dataset_name))
        print("feats shape: ", np.shape(_data))
        print("spker label shape: ", np.shape(_label))
        print("num of spker: ", np.shape(np.unique(_label)))

    def __len__(self):
        return len(self._label)

    def __getitem__(self, index):
        return self._data[index], self._label[index]

    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label

    @property
    def torch_data(self):
        return torch.from_numpy(self._data)

    @property
    def torch_label(self):
        return torch.from_numpy(self._label)

    def sample(self):
        size = len(self.label)
        idx = np.random.randint(0, size-1)
        return self.data[idx], self.label[idx]

    def shuffle(self):
        '''random shuffle data and lable'''
        index = [i for i in range(len(self._label))]
        random.shuffle(index)
        data = self._data[index]
        label = self._label[index]
        return data, label


def get_class_mean(data, label):
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


def get_all_mean(data, label):
    '''compute the all mean and var'''
    all_mean = np.ones((len(np.unique(label)), np.shape(data)[1]), dtype=float)
    all_mean_1d = data.mean(axis=0)
    all_mean = all_mean*all_mean_1d

    return all_mean


if __name__ == "__main__":
    # test
    train_data = feats_data_loader(
        npz_path="./data/feats.npz", dataset_name="vox")
    print(np.shape(get_class_mean(train_data.data, train_data.label)))
    print(np.shape(get_all_mean(train_data.data)))
