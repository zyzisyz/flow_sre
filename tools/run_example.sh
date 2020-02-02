#!/bin/bash

#*************************************************************************
#	> File Name: run_EP_data_visualization.sh
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com 
#	> Created Time: Sun 26 Jan 2020 12:54:07 PM CST
# ************************************************************************/

# convert ark to npz
python -u data/ark2npz.py \
	--src_file data/feats.ark \
	--dest_file data/feats.npz \
	--utt2spk data/utt2spk

# sample and make tsv
python -u tsv_data_prepare.py \
	--npz_path ../test.npz \
    --class_num 30 \
    --sample_num 300 

# sample same index from x and z, make tsv
python -u comp_tsv_data_prepare.py \
	--pre_npz ../data/feats.npz \
	--infered_npz ../test.npz \
	--class_num 30 \
    --sample_num 300 

