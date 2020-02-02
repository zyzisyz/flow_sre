#!/bin/bash

#*************************************************************************
#	> File Name: run.sh
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com 
#	> Created Time: Sat 01 Feb 2020 08:00:01 PM CST
# ************************************************************************/


echo "start to train"
python -u main.py \
	   --flow maf \
	   --epochs 2 \
	   --batch_size 20000 \
	   --train_data_npz ./data/feats.npz \
	   --lr 0.001 \
	   --num_blocks 10 \
	   --num_hidden 256 \
	   --device 0 \
	   --ckpt_dir ckpt \
	   --v_c 0.1 \
	   --v_0 1.0 \
	   --c_dim 36 \
	   --ckpt_save_interval 1

echo ################################################

echo "start to infer data from x space to z space and store to npz"
python -u main.py \
	   --eval \
	   --infer_epoch 1 \
	   --test_data_npz ./data/feats.npz \
	   --infer_data_store_path ./infered.npz


echo ################################################

echo "sample same index(label) data from x space and z space, make tsv"
python -u tools/comp_tsv_data_prepare.py \
    --pre_npz ./data/feats.npz \
    --infered_npz ./infered.npz \
    --class_num 30 \
    --sample_num 300 \
    --tsv_dir ./tsv

