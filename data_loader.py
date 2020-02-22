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



if __name__ == "__main__":
    # test
    train_data = feats_data_loader(
        npz_path="./data/feats.npz", dataset_name="vox")
    print(train_data.data[1])
    print(train_data.label[1])
    print(train_data.utt_label[1])

