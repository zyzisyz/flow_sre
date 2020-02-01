#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: tsv_data_prepare.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Tue 21 Jan 2020 04:12:58 PM CST
# ************************************************************************/

import os
import numpy as np
import random



def save_label_tsv(path, x):
    y = x.flatten()
    n = len(y)
    with open(path, 'w') as f:
        for i in range(n):
            v = y[i]
            f.write('%s\n' % (str(v)))
        print("successfully saved in {}".format(path))


def save_tensor_tsv(path, feats):
    with open(path, 'w') as f:
        num, dim = feats.shape

        for line in range(num):
            for d in range(dim):
                if d > 0:
                    f.write('\t')
                v = str(feats[line][d])
                f.write(v)
            f.write('\n')
        print("successfully saved in {}".format(path))


def write_tsv_embeddings(prefix, feats, labels=None):
    '''
     Write a tensor (or meta) to a tsv file for the `Embedding Project` tool
    :param prefix: output file prefix
    :param feats: embedding tensor NxDim
    :param labels: meta data
    :return: None
    '''
    feat_path = prefix + '_data.tsv'
    save_tensor_tsv(feat_path, feats)
    if labels is None:
        return
    dims = len(labels.shape)
    label_path = prefix + '_meta.tsv'
    if dims == 1:
        save_label_tsv(label_path, labels)
    else:
        save_tensor_tsv(label_path, labels)


def sample(pre_npz, infered_npz, sample_class=50, sample_num=500):
    '''
    :param npz_path: npz file path which stores the datesets
    :param prefix: out_put tsv file name prefix
    :param sample_class: number of sample classes / number of spkers
    :param sample_num: sample number of each spkers
    :return: None
    '''
    print("loading data...")

    pre_data = np.load(pre_npz)['feats']
    pre_label = np.load(pre_npz)['spkers']

    infer_data = np.load(infered_npz)['feats']
    infer_label = np.load(infer_npz)['spkers']

    assert pre_label == infer_label

    '''random shuffle data and lable'''
    index = [i for i in range(len(pre_label))]
    random.shuffle(index)
    pre_data = pre_data[index]
    pre_label = pre_label[index]
    infer_data = infer_data[index]
    infer_label = infer_label[index]

    print("start to sample...")
    sample_index = []
    while(len(sample_index)<sample_class):
        idx = np.random.randint(0, len(np.unique(pre_label))-1)
        if idx not in sample_index:
            sample_index.append(idx)

    sample_pre_data = []
    sample_infer_data = []

    sample_label = []

    print("total label: ", len(_label))
    print("unique label: ", len(np.unique(_label)))
    print("sample data...")
    print("sample class: {}, sample num: {}".format(sample_class, sample_num))

    for idx in sample_index:
        counter = 0
        print("sample idx", idx)
        for i in range(len(_label)):
            if(idx==_label[i]):
                if(counter<sample_num):
                    counter+=1
                    sample_label.append(idx)
                    sample_pre_data.append(pre_data[idx])
                else:
                    break

    
    x = np.array(sample_data)

    y = []
    table = {}
    counter = 0
    for it in sample_label:
        if it not in table:
            table[it] = counter
            y.append(counter)
            counter+=1
        else:
            idx = table[it]
            y.append(idx)

    y = np.array(y)

    write_tsv_embeddings(prefix=prefix, feats=x, labels=y)
    
   

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='a apple a day, keep doctor away')

    parser.add_argument('--npz_path', type=str, default="./data/feats.npz", help='load the npz data')
    parser.add_argument('--class_num',type=int, default=30,help='class num / spker num')
    parser.add_argument('--sample_num',type=int, default=300,help='sample num of each spker')
    args = parser.parse_args()

    # sample, data preparetion and make tsv file
    sample(args.npz_path, args.class_num, args.sample_num)


