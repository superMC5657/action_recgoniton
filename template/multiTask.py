# -*- coding: utf-8 -*-
# !@time: 2021/12/14 上午1:59
# !@author: superMC @email: 18758266469@163.com
# !@fileName: multiTask.py

from __future__ import print_function, division

import copy
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models
# from Classes_Network import *
from torchvision.transforms import transforms

data_dir = 'Dataset/'
train_annotations_file = 'Multi_train_annotation.csv'
val_annotations_file = 'Multi_val_annotation.csv'
CLASSES = ['Mammals', 'Birds']
SPECIES = ['rabbits', 'rats', 'chickens']

pd.read_csv(data_dir + train_annotations_file)


class MyDataset():
    def __init__(self, root_dir, annotations_file, transform=None):
        self.root_dir = root_dir
        self.annotations = annotations_file
        self.transform = transform

        # if not os.path.isfile(self.annotations_file):
        #     print(self.annotations + "does not exist")
        self.file_info = pd.read_csv(root_dir + annotations_file, index_col=0)
        self.size = len(self.file_info)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_path = self.file_info['path'][idx]
        label_classes = self.file_info['classes'][idx]
        label_species = self.file_info['species'][idx]

        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, label_classes, label_species


train_transform = transforms.Compose([transforms.Resize((500, 500)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      ])

# val_transform = transforms.Compose([transforms.Resize((500, 500)),
#                                        transforms.RandomHorizontalFlip(),
#                                        transforms.ToTensor(),
#                                        ])
val_transform = transforms.Compose([transforms.Resize((500, 500)),

                                    transforms.ToTensor(),
                                    ])
train_dataset = MyDataset(data_dir, train_annotations_file, transform=train_transform)
val_dataset = MyDataset(data_dir, val_annotations_file, transform=val_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)

data_loaders = {'train': train_loader, 'val': val_loader}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

model_ft = models.resnet18(pretrained=True)  # 加载已经训练好的模型

# 使除最后一层的参数不可导，即不进行学习
for param in model_ft.parameters():
    param.requires_grad = False

# classes分类结果输出
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 32)  # 将全连接层做出改变类别改为两类


class multi_out_model(torch.nn.Module):
    def __init__(self, model_core):
        super(multi_out_model, self).__init__()

        self.resnet_model = model_core

        self.classes = nn.Linear(in_features=32, out_features=2, bias=True)
        self.species = nn.Linear(in_features=32, out_features=3, bias=True)

    def forward(self, x):
        x1 = self.resnet_model(x)

        classes = self.classes(x1)
        species = self.species(x1)

        return classes, species


model_ft = multi_out_model(model_ft)

criterion = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]

model_ft = model_ft.to(device)
network = model_ft

# criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized优化参数
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Observe that only parameters of final layer are being optimized as
# opoosed to before.

optimizer_ft = optim.SGD([{"params": model_ft.resnet_model.fc.parameters()},
                          {"params": model_ft.classes.parameters()},
                          {"params": model_ft.species.parameters()}], lr=0.01, momentum=0.9)
optimizer = optimizer_ft

# Decay LR by a factor of 0.1 every 7 epochs使用学习率缩减
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


def train_model(model, criterion, optimizer, scheduler, num_epochs=50, pretrain_model=None):
    start_time = time.perf_counter()
    Loss_list = {'train': [], 'val': []}
    classloss_list = {'train': [], 'val': []}
    speciesloss_list = {'train': [], 'val': []}
    Accuracy_list_classes = {'train': [], 'val': []}
    Accuracy_list_species = {'train': [], 'val': []}
    start_epoch = 0

    if pretrain_model != None and os.path.exists(pretrain_model):
        checkpoint = torch.load(pretrain_model)
        model.load_state_dict(checkpoint['models'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        num_epochs = num_epochs + start_epoch
    else:
        print('无保存模型，从头开始训练')

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 100

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-*' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 在train和test的时候。BN层以及Dropout的处理方式不一样,其他都一样，所以没有这两类层的话，可以不进行声明
            else:
                model.eval()

            running_loss = 0.0
            running_classes_loss = 0.0
            running_species_loss = 0.0
            corrects_classes = 0
            correct_species = 0

            # Each epoch has a training and validation phase
            for idx, data in enumerate(data_loaders[phase]):
                img, label_classes, label_species = data
                img = img.to(device)
                label_classes = label_classes.to(device)
                label_species = label_species.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):  # 当是train phase时，以下参数为可导，当为val时，后续包含参数不可导
                    output = model(img)
                    x_classes = output[0]
                    x_species = output[1]
                    #                     x_classes, x_species = models(img)

                    x_classes = x_classes.view(-1, 2)  # 将softmax输出的列向量转换为行向量
                    x_species = x_species.view(-1, 3)

                    _, preds_classes = torch.max(x_classes, 1)  # 输出行向量中最大的元素及其对应的索引值
                    _a, preds_species = torch.max(x_species, 1)
                    # 损失函数，可以依实际情况设定。
                    loss_classes = criterion[0](x_classes, label_classes)
                    loss_species = criterion[1](x_species, label_species)

                    #                     loss = criterion(x_classes, label_classes)  # 单分类时loss函数

                    if phase == 'train':
                        loss = 0.1 * loss_classes + 0.9 * loss_species

                        loss.backward()
                        optimizer.step()  # 进行权值更新

                running_classes_loss += loss_classes.item() * img.size(0)
                running_species_loss += loss_species.item() * img.size(0)

                running_loss += loss.item() * img.size(0)

                corrects_classes += torch.sum(preds_classes == label_classes)
                correct_species += torch.sum(preds_species == label_species)

                epoch_loss = running_loss / len(data_loaders[phase].dataset)
                epoch_class_loss = loss_classes / len(data_loaders[phase].dataset)
                epoch_species_loss = loss_species / len(data_loaders[phase].dataset)

                Loss_list[phase].append(epoch_loss)
                classloss_list[phase].append(epoch_class_loss.cpu().detach().numpy())
                speciesloss_list[phase].append(epoch_species_loss.cpu().detach().numpy())

                epoch_acc_classes = corrects_classes.double() / len(data_loaders[phase].dataset)
                epoch_acc_species = correct_species.double() / len(data_loaders[phase].dataset)
                #             epoch_acc = epoch_acc_classes

                Accuracy_list_classes[phase].append(100 * epoch_acc_classes)
                Accuracy_list_species[phase].append(100 * epoch_acc_species)

                print('{} Loss: {:.4f}  Acc_classes: {:.2%}  Acc_species: {:.2%}'.format(phase, epoch_loss,
                                                                                         epoch_acc_classes,
                                                                                         epoch_acc_species))

                # 更新模型权重及最优准确率
                #             if phase == 'val' and epoch_loss < best_loss:
                if phase == 'val':
                    print('This epoch val loss: {:.4f}'.format(epoch_loss))
                    if epoch_loss < best_loss:
                        # 多任务分类时，仅采用了损失函数进行最优模型的选择，为考虑采用其他指标进行筛选，单一任务时，采用准确率即可。
                        #             if phase == 'val' and epoch_acc > best_acc:
                        #                 best_acc = epoch_acc_classes
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())
                        #                 print('Best val classes Acc: {:.2%}'.format(best_acc))
                        print('Best val loss: {:.4f}'.format(best_loss))

                # 获取模型当前的参数，以便后续继续训练
    pre_state = {'models': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    torch.save(pre_state, 'multi_pre_resnet18_model.pt')

    # 所有epoch结束后，将best_model_wts中的模型参数加载到当前网络中，并保存
    state = {'models': model.load_state_dict(best_model_wts)}
    torch.save(state, 'multi_best_model.pt')

    #     print('Best val classes Acc: {:.2%}'.format(best_acc))
    end_time = time.perf_counter()
    print('训练时间：' + str(end_time - start_time))
    return model, classloss_list, speciesloss_list, Loss_list, Accuracy_list_classes, Accuracy_list_species


start_time = time.perf_counter()
model, classloss_list, speciesloss_list, Loss_list, Accuracy_list_classes, Accuracy_list_species = train_model(
    network,
    criterion,
    optimizer,
    exp_lr_scheduler,
    num_epochs=2)

end_time = time.perf_counter()
print('训练时间：' + str(end_time - start_time))

# models, classloss_list, speciesloss_list, Loss_list, Accuracy_list_classes, Accuracy_list_species = train_model(
#     network, criterion, optimizer, exp_lr_scheduler, num_epochs=20, pretrain_model='multi_pre_resnet18_model.pt')

X1 = range(0, len(Loss_list['train']))
X2 = range(0, len(Loss_list['val']))
Y3 = [i.cpu().numpy() for i in Accuracy_list_classes["train"]]
Y4 = [i.cpu().numpy() for i in Accuracy_list_classes["val"]]
# y3 = Accuracy_list_classes["train"]
# y4 = Accuracy_list_classes["val"]
plt.plot(X1, Y3, color="r", linestyle="-", marker=".", linewidth=1, label="train")
plt.plot(X2, Y4, color="b", linestyle="-", marker=".", linewidth=1, label="val")
plt.ylim(min(min(Y3), min(Y4)) * 0.2, max(max(Y3), max(Y4)) * 1.2)
plt.legend()
plt.title('train and val Classes_acc vs. epoches')
plt.ylabel('Classes_accuracy')
plt.savefig("train and val Classes_acc vs epoches.jpg")

Y5 = [i.cpu().numpy() for i in Accuracy_list_species["train"]]
Y6 = [i.cpu().numpy() for i in Accuracy_list_species["val"]]
# y5 = Accuracy_list_species["train"].cpu().numpy()
# y6 = Accuracy_list_species["val"].cpu().numpy()
plt.plot(X1, Y5, color="r", linestyle="-", marker=".", linewidth=1, label="train")
plt.plot(X2, Y6, color="b", linestyle="-", marker=".", linewidth=1, label="val")
plt.ylim(min(min(Y5), min(Y6)) * 0.2, max(max(Y5), max(Y6)) * 1.2)
plt.legend()
plt.title('train and val Species_acc vs. epoches')
plt.ylabel('Classes_accuracy')
plt.savefig("train and val Species_acc vs epoches.jpg")

Y1 = Loss_list["train"]
Y2 = Loss_list["val"]
Y7 = speciesloss_list['train']
Y8 = speciesloss_list['val']
Y9 = classloss_list['train']
Y10 = classloss_list['val']

plt.plot(X1, Y1, color="b", linestyle="-", marker="o", linewidth=1, label="loss_train")
plt.plot(X2, Y2, color="r", linestyle="-", marker="o", linewidth=1, label="loss_val")

plt.plot(X1, Y7, color="b", linestyle="-", marker="^", linewidth=1, label="specie_loss_train")
plt.plot(X2, Y8, color="r", linestyle="-", marker="^", linewidth=1, label="specie_loss_val")

plt.plot(X1, Y9, color="b", linestyle="-", marker="<", linewidth=1, label="class_loss_train")
plt.plot(X2, Y10, color="r", linestyle="-", marker=">", linewidth=1, label="class_loss_val")

plt.ylim(min(min(Y2), min(Y1)) * (-1.5), max(max(Y2), max(Y1), max(Y8), max(Y7), max(Y10), max(Y9)) * 1.1)

plt.legend()
plt.title('train and val loss vs. epoches')
plt.xlabel("epochs")
plt.ylabel('loss')
plt.savefig("train and val loss vs epoches.jpg")


def visualize_model(model):
    corrects_classes = 0
    corrects_species = 0
    counts = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loaders['val']):
            #             print
            img, label_classes, label_species = data

            #             img = img.to(device)
            label_classes = label_classes.to(device)
            label_species = label_species.to(device)
            #             inputs = Dataset['image']
            #             labels_classes = Dataset['classes'].to(device)

            output = model(img.to(device))
            x_classes = output[0].view(-1, 2)
            _, preds_classes = torch.max(x_classes, 1)
            corrects_classes += torch.sum(preds_classes == label_classes)

            x_species = output[1].view(-1, 3)
            _, preds_species = torch.max(x_species, 1)
            corrects_species += torch.sum(preds_species == label_species)

            torch.cuda.empty_cache()
            plt.figure(figsize=(10, 8))
            plt.imshow(transforms.ToPILImage()(img.squeeze(0)))
            plt.title(
                'predicted classes: {}\n ground-truth classes:{}\n predicted species: {}\n ground-truth species:{}' \
                    .format(CLASSES[preds_classes], CLASSES[label_classes], SPECIES[preds_species],
                            SPECIES[label_species]))

            plt.show()
            counts += 1

        epoch_acc_classes = corrects_classes.double() / counts
        epoch_acc_species = corrects_species.double() / counts

        print("epoch_acc_classes:{} epoch_acc_species:{}".format(epoch_acc_classes, epoch_acc_species))


visualize_model(network)
