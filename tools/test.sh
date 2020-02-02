#!/bin/bash

#*************************************************************************
#	> File Name: test.sh
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com 
#	> Created Time: Thu 30 Jan 2020 01:48:57 PM CST
# ************************************************************************/


# sample same index from x and z, make tsv
python -u comp_tsv_data_prepare.py \
	--pre_npz ../data/feats.npz \
	--infered_npz ../test.npz \
	--class_num 30 \
    --sample_num 300 \
    --tsv_dir ./tsv 

