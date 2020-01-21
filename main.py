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
from trainer import *
from infer import *



if __name__ == "__main__":
	args = get_args()

	flow_trainer = trainer(args)
	flow_trainer.train()
	flow_trainer.save_checkpoint()
	
