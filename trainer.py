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
import kaldi_io
from tqdm import tqdm
from tensorboardX import SummaryWriter

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
        print("torch device: {}".format(self.device))

        # tensorboardX
        self.writer = SummaryWriter(comment=args.model_name)
        self.global_step = 0


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
    
        c_dim = args.c_dim      # hda p dim (default: 36)
        v_c = args.v_c          # variance of the class space (default: 0.1)
        v_0 = args.v_0          # variance of the result space (default: 2.0)
        u_0 = args.u_0          # mean of the result space (default: 1.0)
        DATA_DIM = np.shape(self.dataset.data)[1]

        # init var_global
        var_c = torch.ones(c_dim, device=self.device) * v_c
        var_0 = torch.ones((DATA_DIM-c_dim), device=self.device) * v_0
        var_global = torch.cat((var_c, var_0), 0).unsqueeze(0)
        var_global.to(self.device)

        # init all_mean
        self.all_mean = torch.ones((len(np.unique(self.dataset.label)), DATA_DIM), dtype=float) * u_0

        # main to train
        start_epoch = self.epoch_idx

        if start_epoch >= args.epochs:
            print("training is down!")
            return


        for idx in range(start_epoch, args.epochs):  # epochs
            self.epoch_idx = idx
            train_loss = 0
            train_avg_loss = AverageMeter()

            # init class mean from x space
            if idx == 0:
                self.init_class_mean()
           
            # init contain
            self.contain = []
            for i in range(self.class_mean.shape[0]):
                self.contain.append([])

            pbar = tqdm(total=len(self.train_loader.dataset))
            for batch_idx, (data, label) in enumerate(self.train_loader):  # batchs

                data = data.to(self.device)
                label = label.to(self.device)

                self.optimizer.zero_grad()

                mean_j = torch.index_select(self.class_mean, 0, label).to(self.device)

                '''NOTE: if you want to change loss function, z must be returned to update contain and class mean'''
                # compute hda Gaussion log-likehood
                loss, z = self.model.HDA_Gaussion_log_likehood(data, mean_j, var_global)
                
                # loss
                loss.backward()

                cur_loss = loss.item()

                # loss logging
                self.writer.add_scalar('LogLL', loss.item(), self.global_step)
                self.global_step += 1
                train_loss += cur_loss
                train_avg_loss.update(loss.item(), data.size(0))

                

                self.optimizer.step()
                # update contain
                z = z.cpu().detach().numpy()
                for i in range(len(label)):
                    self.contain[label[i]].append(z[i])

                pbar.update(data.size(0))
                pbar.set_description('[Epoch: {}] HDA Gaussion log-likehood = {:.6f}'.format(idx, train_avg_loss.val))


            pbar.close()
            print('====> Epoch: {} Average loss: {:.6f}'.format(idx, train_avg_loss.avg))

            print("Batch Norm...")
            for module in self.model.modules():
                if isinstance(module, fnn.BatchNormFlow):
                    module.momentum = 0

            # this step need allocate lots of memory...
            print("this step need allocate lots of memory...")
            with torch.no_grad():
                tmp = torch.from_numpy(self.dataset.data[:10000, ]).to(self.device)
                self.model(tmp)

            for module in self.model.modules():
                if isinstance(module, fnn.BatchNormFlow):
                    module.momentum = 1

            # update class mean afer epoch training
            self.update_class_mean()

            if self.epoch_idx % args.ckpt_save_interval == 0:
                self.save_checkpoint()

        self.save_checkpoint()
        print("training is down")


    def init_class_mean(self):
        args = self.args

        DATA_DIM = len(self.dataset.data[0])
        # init class mean from x space at the start of epoch 0
        label_num = len(np.unique(self.dataset.label))

        print('epoch 0: init class_mean by sampling from N~(0, {})'.format(args.v_0))

        mean = np.zeros(DATA_DIM)
        conv = np.identity(DATA_DIM) * args.v_0
        class_mean = np.random.multivariate_normal(mean=mean, cov=conv, size=label_num)
        class_mean = torch.from_numpy(class_mean)

        # split data
        class_mean = class_mean[:, :args.c_dim]
        all_mean = self.all_mean[:, args.c_dim:]

        # cat data
        class_mean = torch.cat((class_mean, all_mean), 1)

        # epoch 0 no u_shift 
        self.class_mean = class_mean.to(self.device)


    def update_class_mean(self):
        args = self.args
        DATA_DIM = len(self.dataset.data[0])
        # update class mean from z space at the start of epoch x > 0 (contain)
        if self.contain != None:
            print('update class_mean from z space and args.u_shift is {}'.format(args.u_shift))
            class_dataset = []
            for it in self.contain:
                class_dataset.append(np.array(it))

            class_mean = np.zeros((len(class_dataset), DATA_DIM))
            for i in range(len(class_dataset)):
                it = class_dataset[i]
                class_mean[i] = it.mean(axis=0)
                
            class_mean = torch.from_numpy(class_mean)

            # split data
            class_mean = class_mean[:, :args.c_dim]
            all_mean = self.all_mean[:, args.c_dim:]

            # cat data
            class_mean = torch.cat((class_mean, all_mean), 1)

            # shift from last epoch class_mean
            self.class_mean = class_mean.to(self.device)*args.u_shift + self.class_mean*(1.0-args.u_shift)
        else:
            raise Exception("contain error")
            

    def generate_z_ark(self):
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
            print("successfully reload {} [model] to infer".format(ckpt_path))

        self.model.to(self.device)
        self.model.eval()

        # init data x
        dataset = feats_data_loader(
            npz_path=self.args.test_data_npz, dataset_name=self.args.dataset_name)
        utt_data = dataset.get_utt_data()

        ark_dir = args.ark_dir
        if not os.path.exists(ark_dir):
            os.makedirs(ark_dir)
        ark_path = args.ark_dir + os.sep + 'feats.ark'

        pbar = tqdm(total=len(utt_data))
        with open(ark_path,'wb') as f:
            for utt, data in utt_data.items():
                data = np.array(data)
                data = torch.from_numpy(data)
                data = data.to(self.device)
                data, _ = self.model(data)
                data = data.cpu().detach().numpy()
                # split data
                data = data[:, :c_dim]
                kaldi_io.write_mat(f, data, utt)
                pbar.update(1)
                pbar.set_description('generate utter {} of frames {}'.format(
                    utt, data.shape[0]))
        pbar.close()
        print("successfully save kaldi ark in {}".format(ark_path))



    def generate_z_np(self):
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
            print("successfully reload {} [model] to infer".format(ckpt_path))

        self.model.to(self.device)
        self.model.eval()

        # init data x
        dataset = feats_data_loader(
            npz_path=self.args.test_data_npz, dataset_name=self.args.dataset_name)

        kwargs = {'num_workers': 6, 'pin_memory': True}
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=25000, shuffle=False, **kwargs)

        pbar = tqdm()
        for batch_idx, (data, label) in enumerate(test_loader):  # batchs
            data = data.to(self.device)
            z, _ = self.model(data)
            z = z.cpu().detach().numpy()
            if batch_idx == 0:
                feats = z
            else:
                feats = np.vstack((feats, z))
            shape = np.shape(feats)
            pbar.set_description('feats.npz generateing: {:.2f}%\t{}'.format(shape[0]/len(dataset.label)*100, shape))

        pbar.close()

        feats = feats[:, :c_dim]
        print(np.shape(feats))
        print("saving...")

        np_dir = args.np_dir
        if not os.path.exists(np_dir):
            os.makedirs(np_dir)
        npz_path = np_dir + os.sep + 'feats.npz'
        np.savez(npz_path, feats=feats, spker_label=dataset.spker_label, utt_label=dataset.utt_label)
        print("sucessfully saved in {}".format(npz_path))

    def reload_checkpoint(self):
        '''check if checkpoint file exists and reload the checkpoint'''
        args = self.args
        self.epoch_idx = 0
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
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
                print("sucessfully reload mdl_epoch{}.pt [optimizer]".format(self.epoch_idx))

                # NOTE: this maybe a bug in pytorch
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

                self.model.load_state_dict(checkpoint_dict['model'])
                print("sucessfully reload mdl_epoch{}.pt [model]".format(self.epoch_idx))

                self.class_mean = checkpoint_dict['class_mean']
                print("sucessfully reload mdl_epoch{}.pt [class_mean]".format(self.epoch_idx))

                print("sucessfully reload mdl_epoch{}.pt all".format(self.epoch_idx))
                self.epoch_idx += 1
                self.contain = None
            else:
                print("start to train fron epoch 0...")

    def save_checkpoint(self):
        '''save the checkpoint, including model and optimizer, and model index is epoch'''
        args = self.args
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
            print("can not find ckpt dir, and creat {} dir".format(args.ckpt_dir))

        PATH = '{}/ckpt_epoch{}.pt'.format(args.ckpt_dir, self.epoch_idx)
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'class_mean': self.class_mean
        }, PATH)


if __name__ == "__main__":
    pass
