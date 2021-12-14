# -*- coding: utf-8 -*-
# !@time: 2021/12/15 上午2:34
# !@author: superMC @email: 18758266469@163.com
# !@fileName: dataset.py
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), ])

val_transform = transforms.Compose([transforms.ToTensor(), ])


class MyDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        self.annotations = annotations_file
        self.transform = transform

        self.file_info = pd.read_csv(annotations_file)
        self.size = len(self.file_info)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        rgb_image_path = self.file_info['rgb_image_path'][idx]
        depth_image_path = self.file_info['depth_image_path'][idx]
        person_label = self.file_info['person_label'][idx]
        sex_label = self.file_info['sex_label'][idx]
        name_label = self.file_info['name_label'][idx]
        action_label = self.file_info['action_label'][idx]

        rgb_image = Image.open(rgb_image_path).convert('RGB')
        depth_image = Image.open(depth_image_path).convert('L')
        if self.transform:
            rgb_image = self.transform(rgb_image)
            depth_image = self.transform(depth_image)
        image = torch.cat([rgb_image, depth_image], dim=0)
        return image, person_label, sex_label, name_label, action_label
