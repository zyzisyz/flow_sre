#!/bin/bash

#*************************************************************************
#	> File Name: train.sh
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
	   --device 0 \
	   --ckpt_dir ckpt \
	   --v_c 0.1 \
	   --v_0 1.0 \
	   --c_dim 36

echo "start to infer z and store to npz"
python -u main.py \
	   --eval \
	   --infer_epoch 0 \
	   --test_data_npz ./data/feats.npz \
	   --infer_data_store_path ./infered.npz

