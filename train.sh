#!/bin/bash

#*************************************************************************
#	> File Name: train.sh
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com 
#	> Created Time: Sat 01 Feb 2020 08:00:01 PM CST
# ************************************************************************/


echo "start to train"
# python -u main.py \
# 	   --epochs 2

echo "infer z and store to npz"
python -u main.py \
	   --epochs 2 \
	   --eval


		
     
