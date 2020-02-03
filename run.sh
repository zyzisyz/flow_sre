#!/bin/bash

#*************************************************************************
#	> File Name: run.sh
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com 
#	> Created Time: Sat 01 Feb 2020 08:00:01 PM CST
# ************************************************************************/



#### config ####
flow=maf
train_data_npz=./data/feats.npz
epochs=2
batch_size=20000 
num_blocks=10 
num_hidden=256 
lr=0.001
v_c=0.1
v_0=1.0
c_dim=36 
ckpt_dir=ckpt/${flow}_block_${num_blocks}_hidden_${num_hidden}_cd_${c_dim}_vc_${v_c}_v0_${v_0}
ckpt_save_interval=1
device=0
kaldi_dir=kaldi_data/${flow}_block_${num_blocks}_hidden_${num_hidden}_cd_${c_dim}_vc_${v_c}_v0_${v_0}


echo "start to train"
python -u main.py \
	   --flow $flow \
	   --epochs $epochs \
	   --batch_size $batch_size \
	   --train_data_npz $train_data_npz \
	   --lr $lr \
	   --num_blocks $num_blocks \
	   --num_hidden $num_hidden \
	   --device $device \
	   --v_c $v_c \
	   --v_0 $v_0 \
	   --c_dim $c_dim \
	   --ckpt_dir $ckpt_dir \
	   --ckpt_save_interval 1

echo "start to infer data from x space to z space and store to kaldi"
python -u main.py \
	   --flow $flow \
	   --eval \
	   --infer_epoch 1 \
	   --kaldi_dir $kaldi_dir

