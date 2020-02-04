#!/bin/bash

#*************************************************************************
#	> File Name: test.sh
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com 
#	> Created Time: Tue 04 Feb 2020 09:53:12 AM CST
# ************************************************************************/



flow=maf
train_data_npz=./data/feats.npz
test_data_npz=./data/feats.npz
epochs=10
batch_size=10000 
num_blocks=10 
num_hidden=256 
lr=0.001
v_c=0.1
v_0=1.0
c_dim=36 
u_0=1.0
u_shift=0.75
ckpt_dir=ckpt/${flow}_block_${num_blocks}_hidden_${num_hidden}_cd_${c_dim}_vc_${v_c}_v0_${v_0}
ckpt_save_interval=2
device=0
kaldi_dir=kaldi_data/${flow}_block_${num_blocks}_hidden_${num_hidden}_cd_${c_dim}_vc_${v_c}_v0_${v_0}

infer_thread_num=3



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
	   --u_shift $u_shift \
	   --u_0 $u_0 \
	   --ckpt_dir $ckpt_dir \
	   --ckpt_save_interval $ckpt_save_interval

