#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: trainer.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Mon 20 Jan 2020 10:52:14 PM CST
# ************************************************************************/

from init_model import *
import os
import torch
from data_loader import *
import torch.optim as optim


class trainer(object):
	def __init__(self, args):
		self.args = args

		# init model
		self.model = init_model(args)

		# init optimizer
		self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

		# init work env
		args.cuda = not args.no_cuda and torch.cuda.is_available()
		if args.cuda:
			os.environ["CUDA_VISIBLE_DEVICES"] = args.device
			self.device = torch.device("cuda:" + args.device)
		else:
			self.device = torch.device("cpu")
		print("training device: {}".format(self.device))

		
	
	def train(self):
		# init dataloader
		self.dataset = feats_data_loader(npz_path="./data/feats.npz", dataset_name="vox")
		self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True)
		start_opoch = self.reload_checkpoint()
		self.model.to(self.device)

		# main to train


	def infer(self):
		pass
	

	def reload_checkpoint(self):
		'''check if checkpoint file exists and reload the checkpoint'''
		args=self.args
		self.epoch=0
		if not os.path.exists(args.ckpt_dir):
			os.mkdir(args.ckpt_dir);
			print("can not find ckpt dir, and creat {} dir".format(args.ckpt_dir))
			print("start to train fron epoch 0...")
		else:
			files = os.listdir(args.ckpt_dir)
			ckpts = []
			for f in files:
				if(f.endswith(".pt")):
					ckpts.append(f)
			if(len(ckpts)):
				import re
				for ckpt in ckpts:
					ckpt_epoch = int(re.findall(r"\d+", ckpt)[0])
					if ckpt_epoch>self.epoch:
						self.epoch=ckpt_epoch

				checkpoint_dict = torch.load('{}/ckpt_epoch{}.pt'.format(args.ckpt_dir, self.epoch), map_location=self.device)
				self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
				self.model.load_state_dict(checkpoint_dict['model'])
				print("sucessfully reload mdl_epoch{}.pt".format(self.epoch))
				self.epoch+=1

		return self.epoch

		
	def save_checkpoint(self):
		'''save the checkpoint, including model and optimizer, and model index is epoch'''
		args=self.args
		if not os.path.exists(args.ckpt_dir):
			os.mkdir(args.ckpt_dir);
			print("can not find ckpt dir, and creat {} dir".format(args.ckpt_dir))

		PATH = '{}/ckpt_epoch{}.pt'.format(args.ckpt_dir, self.epoch)
		torch.save({
			'model': self.model.state_dict(),
			'optimizer': self.optimizer.state_dict()
			}, PATH)

