#!/bin/bash

#*************************************************************************
#	> File Name: run.sh
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com 
#	> Created Time: Sat 01 Feb 2020 08:00:01 PM CST
# ************************************************************************/


##################################################################
# stage0: Parameter Config
##################################################################

flow=maf
train_data_npz=./data/feats.npz
test_data_npz=./data/feats.npz
epochs=10
batch_size=20000 
num_blocks=10 
num_hidden=256 
lr=0.001
v_c=0.1
v_0=1.0
c_dim=36 
u_0=1.0
u_shift=0.75
ckpt_dir=ckpt/${flow}_block_${num_blocks}_hidden_${num_hidden}_cd_${c_dim}_vc_${v_c}_v0_${v_0}_u0_${u_0}
ckpt_save_interval=2
device=0
kaldi_dir=kaldi_data/${flow}_block_${num_blocks}_hidden_${num_hidden}_cd_${c_dim}_vc_${v_c}_v0_${v_0}_u0_${u_0}
log_dir=log_data/${flow}_block_${num_blocks}_hidden_${num_hidden}_cd_${c_dim}_vc_${v_c}_v0_${v_0}_u0_${u_0}

infer_thread_num=3


##################################################################
# stage2: pytorch training 
##################################################################

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
	   --u_0 $u_0 \
	   --u_shift $u_shift \
	   --c_dim $c_dim \
	   --ckpt_dir $ckpt_dir \
	   --ckpt_save_interval $ckpt_save_interval

##################################################################
# stage2: infer data from x space to z space and save to kaldi ark
##################################################################

echo "start to infer data from x space to z space and store to kaldi"
for ((infer_epoch=0;infer_epoch<${epochs};infer_epoch=infer_epoch+ckpt_save_interval))
do
	python -u main.py \
		--eval \
		--test_data_npz $test_data_npz \
		--flow $flow \
		--num_blocks $num_blocks \
		--num_hidden $num_hidden \
		--infer_epoch $infer_epoch \
		--device $device \
		--ckpt_dir $ckpt_dir \
		--kaldi_dir $kaldi_dir
done 

tempfifo="temp_fifo"
mkfifo ${tempfifo}

exec 6<>${tempfifo}
rm -f ${tempfifo}

for ((i=1;i<=${thread_num};i++))
do
{
    echo 
}
done >&6 


for ((infer_epoch=0;infer_epoch<${epochs};infer_epoch=infer_epoch+ckpt_save_interval))
do
	{
		nohup python -u main.py \
			--eval \
			--test_data_npz $test_data_npz \
			--flow $flow \
			--num_blocks $num_blocks \
			--num_hidden $num_hidden \
			--infer_epoch $infer_epoch \
			--device $device \
			--ckpt_dir $ckpt_dir \
			--kaldi_dir $kaldi_dir \
		> $log_dir/infer_log_${infer_epoch}.log

		echo "" >&6
	} & 
done 

# close fd6 pipline (fifo)
exec 6>&-

