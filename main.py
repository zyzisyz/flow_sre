#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: main.py
#	> Author: Yang Zhang
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Mon 20 Jan 2020 10:42:06 PM CST
# ************************************************************************/


from utils import get_args
from trainer import trainer


if __name__ == "__main__":
    args = get_args()
    flow_trainer = trainer(args)
    if not args.eval:
        flow_trainer.train()
    else:
        if not args.ark:
            flow_trainer.generate_z_np()
        else:
            flow_trainer.generate_z_kaidl()
