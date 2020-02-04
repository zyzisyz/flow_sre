#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: npz2ark.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Tue 04 Feb 2020 04:07:38 PM CST
# ************************************************************************/

import numpy as np
import kaldi_io
from tqdm import tqdm
import os

def npz2ark(npz_path, kaldi_dir):
    print("loading...")
    feats = np.load(npz_path)['feats']
    utt_label = np.load(npz_path)['utt_label']
    utt_class = np.unique(utt_label)
    utt_data = {}
    for i in utt_class:
        utt_data[i] = []
    for i in range(len(feats)):
        key = utt_label[i]
        utt_data[key].append(feats[i])

    if not os.path.exists(kaldi_dir):
        os.makedirs(kaldi_dir)
    ark_path = kaldi_dir + os.sep + 'feats.ark'

    print("ark writing...")
    pbar = tqdm(total=len(utt_data))
    with open(ark_path,'wb') as f:
        for utt, data in utt_data.items():
            data = np.array(data)
            kaldi_io.write_mat(f, data, utt)
            pbar.update(1)
            pbar.set_description('generate utter {} of frames {}'.format(utt, data.shape[0]))
    pbar.close()
    print("successfully save kaldi ark in {}".format(ark_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_file', default="feats.npz", help='src file of feats.(npz)')
    parser.add_argument('--dest_dir', default="./", help='dest dir of feats.(npz)')
    args = parser.parse_args()

    npz2ark(args.src_file, args.dest_dir)

