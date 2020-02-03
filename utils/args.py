#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: args.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Mon 20 Jan 2020 10:44:10 PM CST
# ************************************************************************/

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Flows for sre')

    parser.add_argument(
            '--batch_size',
            type=int,
            default=20000,
            help='input batch size for training (default: 20000)')

    parser.add_argument(
            '--epochs',
            type=int,
            default=10,
            help='number of epochs to train (default: 100)')

    parser.add_argument(
            '--infer_epoch',
            type=int,
            default=-1,
            help='index of ckpt epoch to infer (default: 100)')

    parser.add_argument(
            '--num_hidden',
            type=int,
            default=256,
            help='number of hidden of model')

    parser.add_argument(
            '--lr', type=float, default=0.001, help='learning rate (default: 0.001)')

    parser.add_argument(
            '--flow', default='maf', help='flow to use: maf | realnvp | glow')

    parser.add_argument(
            '--no_cuda',
            action='store_true',
            default=False,
            help='disables CUDA training')

    parser.add_argument(
            '--np',
            action='store_true',
            default=False,
            help='infer kaldi or numpy data')

    parser.add_argument(
            '--num_blocks',
            type=int,
            default=10,
            help='number of invertible blocks (default: 10)')

    parser.add_argument(
            '--seed', type=int, default=1, help='random seed (default: 1)')

    parser.add_argument(
            '--device',
            default='0',
            help='cuda visible devices (default: 0)')

    parser.add_argument(
            '--log',
            default='log',
            help='log status')

    parser.add_argument(
            '--kaldi_dir',
            default='infer_kaldi_data',
            help='kaldi ark data')

    parser.add_argument(
            '--train_data_npz',
            default='./data/feats.npz',
            help='train data npz path')

    parser.add_argument(
            '--dataset_name',
            default='voxceleb1',
            help='dataset name')

    parser.add_argument(
            '--infer_data_store_path',
            default='./infer.npz',
            help='infer data npz path')

    parser.add_argument(
            '--test_data_npz',
            default='./data/feats.npz',
            help='infer data npz path')

    parser.add_argument(
            '--ckpt_save_interval',
            type=int,
            default=1,
            help='how many epochs to wait before saving models')

    parser.add_argument(
            '--ckpt_dir',
            default='ckpt',
            help='dir to save check points')

    parser.add_argument(
            '--eval',
            action='store_true',
            default=False,
            help='disables CUDA training')

    parser.add_argument('--v_c', type=float, default=0.1, help='variance of the class space (default: 0.1)')
    parser.add_argument('--v_0', type=float, default=1.0, help='variance of the result space (default: 2.0)')
    parser.add_argument('--c_dim', type=int, default=36, help='variance of the result space (default: 2.0)')

    args = parser.parse_args()

    return args

