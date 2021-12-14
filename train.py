# -*- coding: utf-8 -*-
# !@time: 2021/12/15 上午2:36
# !@author: superMC @email: 18758266469@163.com
# !@fileName: train.py
import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from config import train_csv_path, val_csv_path
from dataset import MyDataset, train_transform, val_transform
from models.actionRecNet import ActionRecNet
from utils import train, validate, adjust_learning_rate, save_checkpoint

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
args = parser.parse_args()

train_dataset = MyDataset(train_csv_path, transform=train_transform)
val_dataset = MyDataset(val_csv_path, transform=val_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)

actionNet = ActionRecNet()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args.lr = 0.01
args.device = device
args.print_freq = 100

# device = torch.device("cpu")
print(device)
criterion = nn.CrossEntropyLoss(reduce=False)

actionNet.to(device)

optimizer = optim.SGD(actionNet.parameters(), lr=0.01, momentum=0.9)

epochs = 50
best_acc1 = 0.0
for epoch in range(epochs):
    adjust_learning_rate(optimizer, epoch, args)
    train(train_loader, actionNet, criterion, optimizer, epoch, args)
    acc1 = validate(val_loader, actionNet, criterion, args)
    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)
    if is_best:
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': actionNet.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)
