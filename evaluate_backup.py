import os.path

from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import Dataset
import random
import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary
import utils
import json
import time


use_DataParallel = torch.cuda.device_count() > 1
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss



class Evaluate:
    def __init__(self, batch_size):
        self.drop_path_prob = None
        self.dataset = Dataset()
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.train_loader, self.valid_loader, self.test_loader, self.classes = self.dataset.get_dataset(self.batch_size)
        self.lr = 0.001

    def __train_epoch(self, train_queue, model, criterion, optimizer, epoch):
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        model.train()

        with tqdm(train_queue) as progress:
            progress.set_description_str(f'Train epoch {epoch}')

            for step, (x, target) in enumerate(progress):

                x, target = x.to(device), target.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits, logits_aux = model(x)
                loss = criterion(logits, target)
                # if self.__args.auxiliary:
                #     loss_aux = criterion(logits_aux, target)
                #     loss += self.__args.auxiliary_weight * loss_aux
                loss.backward()
                nn.utils.clip_grad_norm_(model.module.parameters() if use_DataParallel else model.parameters(),5)
                optimizer.step()

                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = x.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

                progress.set_postfix_str(f'loss: {objs.avg}, top1: {top1.avg}')

                print(f'Step:{step:03} loss:{objs.avg} acc1:{top1.avg} acc5:{top5.avg}')

        return top1.avg, objs.avg

    def __infer_epoch(self, valid_queue, model, criterion, epoch):
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        model.eval()

        with tqdm(valid_queue) as progress:
            for step, (x, target) in enumerate(progress):
                progress.set_description_str(f'Valid epoch {epoch}')

                x = x.to(device)
                target = target.to(device, non_blocking=True)

                with torch.no_grad():
                    logits, _ = model(x)
                    loss = criterion(logits, target)

                    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                    n = x.size(0)
                    objs.update(loss.item(), n)
                    top1.update(prec1.item(), n)
                    top5.update(prec5.item(), n)

                progress.set_postfix_str(f'loss: {objs.avg}, top1: {top1.avg}')

                print(f'>> Validation: {step:03} {objs.avg} {top1.avg} {top5.avg}')

        return top1.avg, top5.avg, objs.avg

    def train(self, model, epochs,hash_indv,drop_path_prob = 0.2,warmup=False):
        if use_DataParallel:
            print('use Data Parallel')
            model = nn.DataParallel(model)
            self.__module = model.module
            torch.cuda.manual_seed_all(2)
        else:
            torch.cuda.manual_seed(2)
        model.to(device)
        num_workers = torch.cuda.device_count() * 4

        pytorch_total_params = sum(p.numel() for p in model.parameters())
        pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.optimizer = optim.SGD(model.parameters(), lr=.01,
                      momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        best_acc_top1 = 0
        for epoch in tqdm(range(epochs), desc='Total Progress'):
            self.scheduler.step()
            print(f'epoch {epoch} lr {self.scheduler.get_lr()[0]}')
            self.drop_path_prob = drop_path_prob * epoch / epochs

            train_acc, train_obj = self.__train_epoch(self.train_loader, model, self.criterion,
                                                      self.optimizer, epoch + 1)
            print(f'train_acc: {train_acc}')

            valid_acc_top1, valid_acc_top5, valid_obj = self.__infer_epoch(self.valid_loader, model,
                                                                           self.criterion, epoch + 1)
            print(f'valid_acc: {valid_acc_top1}')


            is_best = False
            if valid_acc_top1 > best_acc_top1:
                best_acc_top1 = valid_acc_top1
                is_best = True


        print(train_acc)
        print(best_acc_top1)
        print(valid_acc_top1)
        return 100 - valid_acc_top1

