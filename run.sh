#!/bin/bash

#*************************************************************************
#	> File Name: run.sh
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com 
#	> Created Time: Sat 01 Feb 2020 08:00:01 PM CST
# ************************************************************************/

#./data_preparation.sh

##################################################################
# stage0: Model Parameter Config
##################################################################

flow=maf
train_data_npz=./data/feats.npz
test_data_npz=./data/feats.npz
epochs=11
batch_size=20000 
num_blocks=10 
num_hidden=512
lr=0.001
v_c=0.1
v_0=1.0
c_dim=36 
u_0=1.0
u_shift=0.75
model_name=${flow}_block_${num_blocks}_hidden_${num_hidden}_cd_${c_dim}_vc_${v_c}_v0_${v_0}_u0_${u_0}
ckpt_dir=ckpt/${model_name}
ckpt_save_interval=1
device=0
infer_data_dir=infered_data/${model_name}
log_dir=log_data/${model_name}

infer_job_num=4

##################################################################
# stage1: pytorch training 
##################################################################

echo "start to train"
python -u main.py \
	   --flow $flow \
	   --epochs $epochs \
	   --batch_size $batch_size \
	   --train_data_npz $train_data_npz \
	   --lr $lr \
	   --model_name $model_name \
	   --num_blocks $num_blocks \
	   --num_hidden $num_hidden \
	   --device $device \
	   --v_c $v_c \
	   --v_0 $v_0 \
	   --u_0 $u_0 \
	   --u_shift $u_shift \
	   --c_dim $c_dim \
	   --ckpt_dir $ckpt_dir \
	   --ckpt_save_interval $ckpt_save_interval


# tensorboard --logdir runs/*

##################################################################
# stage3: infer data from x space to z space and save to numpy npz
##################################################################

echo "start to infer data from x space to z space and store to numpy npz"
for ((infer_epoch=0;infer_epoch<${epochs};infer_epoch=infer_epoch+ckpt_save_interval))
do
	np_dir=$infer_data_dir/$infer_epoch
	python -u main.py \
		--eval \
		--test_data_npz $test_data_npz \
		--flow $flow \
		--num_blocks $num_blocks \
		--num_hidden $num_hidden \
		--infer_epoch $infer_epoch \
		--device $device \
		--ckpt_dir $ckpt_dir \
		--np_dir $np_dir 
done

##################################################################
# stage4: parallel transfer numpy npz to kaldi ark
##################################################################


tempfifo="temp_fifo"
mkfifo ${tempfifo}

exec 6<>${tempfifo}
rm -f ${tempfifo}

for ((i=1;i<=${infer_job_num};i++))
do
{
    echo 
}
done >&6 


for ((infer_epoch=0;infer_epoch<${epochs};infer_epoch=infer_epoch+ckpt_save_interval))
do
	read -u6
	{
		np_file=$infer_data_dir/$infer_epoch/feats.npz
		ark_dir=$infer_data_dir/$infer_epoch

		python -u tools/npz2ark.py \
			--src_file $np_file \
			--dest_dir $ark_dir

		echo "" >&6
	} & 
done 

# close fd6 pipline (fifo)
exec 6>&-

