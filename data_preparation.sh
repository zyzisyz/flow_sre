#!/bin/bash

#*************************************************************************
#	> File Name: data_preparation.sh
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com 
#	> Created Time: Wed 05 Feb 2020 01:03:22 PM CST
# ************************************************************************/


# convert ark to npz
python -u tools/ark2npz.py \
    --src_file data/feats.ark \
    --dest_file data/feats.npz \
    --utt2spk data/utt2spk

