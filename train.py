import argparse
import copy
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
from utils import AverageMeter

import datasets
import flows as fnn
from utils import *

if sys.version_info < (3, 6):
    print('Sorry, this code might need Python 3.6 or higher')

# settings
parser = argparse.ArgumentParser(description='PyTorch Flows')
parser.add_argument(
    '--batch-size',
    type=int,
    default=10000,
    help='input batch size for training (default: 10000)')
parser.add_argument(
    '--epochs',
    type=int,
    default=100,
    help='number of epochs to train (default: 100)')
parser.add_argument(
    '--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
parser.add_argument(
    '--dataset',
    default='VOXCELEB1',
    help='VOXCELEB1')
parser.add_argument(
    '--flow', default='maf', help='flow to use: maf | realnvp | glow')
parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')
parser.add_argument(
    '--num-blocks',
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
    '--interval',
    type=int,
    default=10,
    help='how many epochs to wait before saving models')
parser.add_argument(
    '--chkpt',
    default='chkpt',
    help='dir to save check points')

args = parser.parse_args()

# work env
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda:" + args.device)
    # device = torch.cuda.set_device(int(args.device))
else:
    device = torch.device("cpu")

#device = "cuda"
print("training device: {}".format(device))

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
kwargs = {'num_workers': 6, 'pin_memory': True} if args.cuda else {}

assert args.dataset in ['VOXCELEB1']
dataset = getattr(datasets, args.dataset)()

# load data
train_tensor = torch.from_numpy(dataset.data.x)
train_dataset = torch.utils.data.TensorDataset(train_tensor)
num_cond_inputs = None
 
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

# network structure
num_inputs = dataset.n_dims
num_hidden = {
    'VOXCELEB1': 256
}[args.dataset]

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

model.to(device)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# tensorboardX
writer = SummaryWriter(comment=args.flow + "_" + args.dataset)
global_step = 0

# train
def train(epoch):
    global global_step, writer
    model.train()

    # loss manager
    train_avg_loss = AverageMeter()
    train_log_probs_avg_loss = AverageMeter()
    train_log_jacob_avg_loss = AverageMeter()

    f = open(args.log, 'a')

    # tqdm
    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None
                data = data[0]

        data = data.to(device)
        optimizer.zero_grad()
        loss, loss_probs, loss_jacob = model.log_probs(data, cond_data)
        loss.backward()
        optimizer.step()

        train_avg_loss.update(loss.item(), data.size(0))
        train_log_probs_avg_loss.update(loss_probs.item(), data.size(0))
        train_log_jacob_avg_loss.update(loss_jacob.item(), data.size(0))
        pbar.update(data.size(0))
        pbar.set_description('LogLL = {:.6f}  LogP = {:.6f}  LogDet = {:.6f}'.format(
            train_avg_loss.val, train_log_probs_avg_loss.val, train_log_jacob_avg_loss.val))

        writer.add_scalar('LogLL', loss.item(), global_step)
        writer.add_scalar('LogP', loss_probs.item(), global_step)
        writer.add_scalar('LogDet', loss_jacob.item(), global_step)

        # average loss in this mini-batch
        f.write('LogLL = {:.6f}  LogP = {:.6f}  LogDet = {:.6f}\n'.format(
            train_avg_loss.val, train_log_probs_avg_loss.val, train_log_jacob_avg_loss.val))

        global_step += 1
        
    pbar.close()
    f.close()   
 
    # average loss in this epoch
    print('\nEpoch {} : LogLL = {:.6f}  LogP = {:.6f}  LogDet = {:.6f}'.format(
         epoch, train_avg_loss.avg, train_log_probs_avg_loss.avg, train_log_jacob_avg_loss.avg))

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 0

    # this step need allocate lots of memory...
    with torch.no_grad():
        #model(train_loader.dataset.tensors[0].to(data.device))
        model(train_loader.dataset.tensors[0][:1000].to(data.device))

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 1


#------------#
#### Main ####
#------------#
print("\nTraining started..")
print(time.asctime( time.localtime(time.time()) ))

# check if mdl exists and reload the mdl
start_epoch=0
if not os.path.exists(args.chkpt):
	os.mkdir(args.chkpt);
	print("can not find chkpt dir, and creat {} dir".format(args.chkpt))
else:
	files = os.listdir(args.chkpt)
	chkpts = []
	for f in files:
		if(f.endswith(".pt")):
			chkpts.append(f)
	if(len(chkpts)):
		import re
		for chkpt in chkpts:
			chkpt_epoch = int(re.findall(r"\d+", chkpt)[0])
			if chkpt_epoch>start_epoch:
				start_epoch=chkpt_epoch
		model.load_state_dict(torch.load('{}/mdl_epoch{}.pt'.format(args.chkpt, start_epoch)))
		print("reload mdl_epoch{}.pt".format(start_epoch))
		start_epoch+=1

for epoch in range(start_epoch, args.epochs):
    print('\nEpoch {}'.format(epoch))
    print(time.asctime( time.localtime(time.time()) ))

    train(epoch)
 
    # save model to checkpoint
    if epoch % args.interval == 0:
        print("Saving model to {}/mdl_epoch{}.pt".format(args.chkpt, epoch))
        if not os.path.exists(args.chkpt):
            os.mkdir(args.chkpt);
        torch.save(model.state_dict(), '{}/mdl_epoch{}.pt'.format(args.chkpt, epoch))
    print(time.asctime( time.localtime(time.time()) ))

print('\n' + time.asctime( time.localtime(time.time()) ))
print("Training end..")

