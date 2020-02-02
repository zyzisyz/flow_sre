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
from utils import *
import torch.nn.functional as F

pi = torch.from_numpy(np.array(np.pi))


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
        args = self.args

        kwargs = {'num_workers': 7, 'pin_memory': True}

        # init dataloader
        self.dataset = feats_data_loader(
            npz_path=args.train_data_npz, dataset_name=args.dataset_name)

        self.train_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.args.batch_size, shuffle=True, **kwargs)

        self.reload_checkpoint()
        self.model.to(self.device)

        c_dim = args.c_dim
        v_c = args.v_c
        v_0 = args.v_0

        # main to train
        start_epoch = self.epoch_idx
        for idx in range(start_epoch, args.epochs):  # epochs
            self.epoch_idx = idx
            train_loss = 0

            # firstly, we udpate class_mean
            if idx == 0:
                print('epoch 0: init mean from original dataset')
                class_mean = get_class_mean(
                    self.dataset.data, self.dataset.label)
                class_mean = torch.from_numpy(class_mean)

                all_mean = get_all_mean(self.dataset.data, self.dataset.label)
                all_mean = torch.from_numpy(all_mean)

                # split data
                class_mean = class_mean[:, :c_dim]
                all_mean = all_mean[:, c_dim:]

                # cat data
                class_mean = torch.cat((class_mean, all_mean), 1)
                self.class_mean = class_mean.to(self.device)

            elif self.contain != None:
                print('epoch {}: update mean from z space'.format(idx))
                DATA_DIM = np.shape(self.dataset.data)[1]

                class_dataset = []
                all_mean_1d = np.zeros(DATA_DIM, dtype=float)
                for it in self.contain:
                    class_dataset.append(np.array(it))
                    for i in it:
                        all_mean_1d += i

                all_mean_1d = all_mean_1d/len(self.dataset.label)
                all_mean = np.ones((len(class_dataset), DATA_DIM), dtype=float)
                all_mean = all_mean*all_mean_1d

                class_mean = np.zeros((len(class_dataset), DATA_DIM))

                for i in range(len(class_dataset)):
                    it = class_dataset[i]
                    class_mean[i] = it.mean(axis=0)

                class_mean = torch.from_numpy(class_mean)
                all_mean = torch.from_numpy(all_mean)

                # split data
                class_mean = class_mean[:, :c_dim]
                all_mean = all_mean[:, c_dim:]

                # cat data
                class_mean = torch.cat((class_mean, all_mean), 1)
                self.class_mean = class_mean.to(self.device)

            # init contain
            self.contain = []
            for i in range(self.class_mean.shape[0]):
                self.contain.append([])

            for batch_idx, (data, label) in enumerate(self.train_loader):  # batchs

                for i in range(len(label)):
                    self.contain[label[i]].append(data[i].numpy())

                data = data.to(self.device)
                label = label.to(self.device)
                self.class_mean.to(self.device)

                self.optimizer.zero_grad()

                # init var_global
                var_c = torch.ones(
                    data[:, :c_dim].shape[1], device=data.device) * v_c
                var_0 = torch.ones(
                    data[:, c_dim:].shape[1], device=data.device) * v_0
                var_global = torch.cat((var_c, var_0), 0).unsqueeze(0)
                var_global.to(self.device)

                # convert data to z space
                z, logdet = self.model(data)

                # compute hda Guassion log-likehood
                mean_j = torch.index_select(self.class_mean, 0, label)

                log_det_sigma = torch.log(
                    var_global+1e-15).sum(-1, keepdim=True).to(self.device)
                log_probs = -0.5 * ((torch.pow((z-mean_j), 2) / (var_global+1e-15) + torch.log(
                    2 * pi)).sum(-1, keepdim=True) + log_det_sigma).to(self.device)

                loss = -(log_probs + logdet).mean()

                # loss
                loss.backward()

                cur_loss = loss.item()

                train_loss += cur_loss
                self.optimizer.step()

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.epoch_idx, batch_idx *
                    len(data), len(self.train_loader.dataset),
                    100.*batch_idx / len(self.train_loader),
                    cur_loss))

            # print('====> Epoch: {} Average loss: {:.4f}'.format(
            #    self.epoch_idx, train_loss)/len(self.label))

            if self.epoch_idx % args.ckpt_save_interval == 0:
                self.save_checkpoint()

        self.save_checkpoint()

    # generate z

    def generate_z(self):
        args = self.args

        c_dim = self.args.c_dim

        # init model
        if args.infer_epoch == -1:
            self.reload_checkpoint()
        else:
            ckpt_path = '{}/ckpt_epoch{}.pt'.format(args.ckpt_dir, args.infer_epoch)
            assert os.path.exists(ckpt_path) == True
            checkpoint_dict = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint_dict['model'])
            print("successfully reload {} to infer".format(ckpt_path))

        self.model.to(self.device)
        self.model.eval()

        # init data x
        dataset = feats_data_loader(
            npz_path=self.args.test_data_npz, dataset_name=self.args.dataset_name)
        labels = dataset.label

        kwargs = {'num_workers': 6, 'pin_memory': True}
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=25000, shuffle=False, **kwargs)

        total = len(dataset.label)
        for batch_idx, (data, label) in enumerate(test_loader):  # batchs
            data = data.to(self.device)
            self.optimizer.zero_grad()
            z, _ = self.model(data)
            z = z.cpu().detach().numpy()
            if batch_idx == 0:
                feats = z
            else:
                feats = np.vstack((feats, z))

            print(batch_idx, " generating: {:.2f}% \t[{}/{}]\tfeats shape: {}".format(
                (batch_idx+1)*len(label)/total*100, (batch_idx+1)*len(label), total, np.shape(feats)))

        feats = feats[:, :c_dim]
        print(np.shape(feats))

        np.savez(args.infer_data_store_path, feats=feats, spkers=labels)
        print("sucessfully saved in {}".format(args.infer_data_store_path))

    def reload_checkpoint(self):
        '''check if checkpoint file exists and reload the checkpoint'''
        args = self.args
        self.epoch_idx = 0
        if not os.path.exists(args.ckpt_dir):
            os.mkdir(args.ckpt_dir)
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
                    if ckpt_epoch > self.epoch_idx:
                        self.epoch_idx = ckpt_epoch

                checkpoint_dict = torch.load(
                    '{}/ckpt_epoch{}.pt'.format(args.ckpt_dir, self.epoch_idx), map_location=self.device)
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])

                # NOTE: this maybe a bug in pytorch
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

                self.model.load_state_dict(checkpoint_dict['model'])
                self.class_mean = checkpoint_dict['class_mean']

                print("sucessfully reload mdl_epoch{}.pt".format(self.epoch_idx))
                self.epoch_idx += 1
                self.contain = None
            else:
                print("start to train fron epoch 0...")

    def save_checkpoint(self):
        '''save the checkpoint, including model and optimizer, and model index is epoch'''
        args = self.args
        if not os.path.exists(args.ckpt_dir):
            os.mkdir(args.ckpt_dir)
            print("can not find ckpt dir, and creat {} dir".format(args.ckpt_dir))

        PATH = '{}/ckpt_epoch{}.pt'.format(args.ckpt_dir, self.epoch_idx)
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'class_mean': self.class_mean
        }, PATH)


if __name__ == "__main__":
    pass
