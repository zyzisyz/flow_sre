#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: _init_model.py
#	> Author: Yang Zhang
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Mon 20 Jan 2020 11:19:38 PM CST
# ************************************************************************/

import flows as fnn
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_model(args, num_inputs=72):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        device = torch.device("cuda:" + args.device)
    else:
        device = torch.device("cpu")
    # network structure
    num_hidden = args.num_hidden
    num_cond_inputs = None

    act = 'relu'
    assert act in ['relu', 'sigmoid', 'tanh']

    modules = []

    # normalization flow
    assert args.flow in ['maf', 'realnvp', 'glow']

    if args.flow == 'glow':
        mask = torch.arange(0, num_inputs) % 2
        mask = mask.to(device).float()

        print("Warning: Results for GLOW are not as good as for MAF yet.")
        for _ in range(args.num_blocks):
            modules += [
                fnn.BatchNormFlow(num_inputs),
                fnn.LUInvertibleMM(num_inputs),
                fnn.CouplingLayer(
                    num_inputs, num_hidden, mask, num_cond_inputs,
                    s_act='tanh', t_act='relu')
            ]
            mask = 1 - mask

    elif args.flow == 'realnvp':
        mask = torch.arange(0, num_inputs) % 2
        mask = mask.to(device).float()

        for _ in range(args.num_blocks):
            modules += [
                fnn.CouplingLayer(
                    num_inputs, num_hidden, mask, num_cond_inputs,
                    s_act='tanh', t_act='relu'),
                fnn.BatchNormFlow(num_inputs)
            ]
            mask = 1 - mask

    elif args.flow == 'maf':
        for _ in range(args.num_blocks):
            modules += [
                fnn.MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
                fnn.BatchNormFlow(num_inputs),
                fnn.Reverse(num_inputs)
            ]

    model = fnn.FlowSequential(*modules)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.fill_(0)

    return model
