#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: main.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Mon 20 Jan 2020 10:42:06 PM CST
# ************************************************************************/


import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from tensorboardX import SummaryWriter


from utils import *
from init_model import *



if __name__ == "__main__":
	args = get_args()

	# work env
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	if args.cuda:
		os.environ["CUDA_VISIBLE_DEVICES"] = args.device
		device = torch.device("cuda:" + args.device)
	else:
		device = torch.device("cpu")

	# model init
	model = init_model(args)	
	model.to(device)

	# data preparation
	dataset = feats_data_loader(npz_path="./data/feats.npz", dataset_name="vox")
	loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch-size, shuffle=True)

	if(args.infer==False):
		trainer(model=model, train_loader=loader)
	else:
		infer(model=model, test_loader=loader, infer_epoch=args.infer_epoch)

