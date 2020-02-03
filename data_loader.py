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
    def __init__(self, npz_path="data/feats.npz", dataset_name="voxceleb1"):
        assert os.path.exists(npz_path) == True
        print("hi")

        self.dataset_name = dataset_name

        feat_data = np.load(npz_path)['feats']
        spker_label = np.load(npz_path)['spker_label'] # spker label for pytorch train
        utt_label = np.load(npz_path)['utt_label'] # utt label for kaili test

        self.feat_data = feat_data
        self.spker_label = spker_label
        self.utt_label = utt_label

        print("dataset: {}".format(dataset_name))
        print("feats shape: ", np.shape(feat_data))
        print("spker label shape: ", np.shape(spker_label))
        print("num of spker: ", np.shape(np.unique(spker_label)))
        print("utt label shape: ", np.shape(utt_label))
        print("num of utt: ", np.shape(np.unique(utt_label)))

    def get_utt_data(self):
        '''for kaldi test'''
        utt_class = np.unique(self.utt_label)
        utt_data = {}
        for i in utt_class:
            utt_data[i] = []
        for i in range(len(self.data)):
            key = self.utt_label[i]
            utt_data[key].append(self.data[i])
        
        return utt_data

    def __len__(self):
        return len(self.spker_label)

    def __getitem__(self, index):
        return self.feat_data[index], self.spker_label[index]

    @property
    def data(self):
        return self.feat_data

    @property
    def label(self):
        '''default label is spker_label'''
        return self.spker_label


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


if __name__ == "__main__":
    # test
    train_data = feats_data_loader(
        npz_path="./data/feats.npz", dataset_name="vox")
    utt_data = train_data.utt_data()

