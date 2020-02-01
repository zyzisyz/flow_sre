#!/bin/bash

#*************************************************************************
#	> File Name: test.sh
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com 
#	> Created Time: Thu 30 Jan 2020 01:48:57 PM CST
# ************************************************************************/


# sample and make tsv
python -u tsv_data_prepare.py \
	--npz_path ../test.npz \
    --class_num 100 \
    --sample_num 300 
