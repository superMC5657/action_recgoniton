# -*- coding: utf-8 -*-
# !@time: 2021/12/15 上午3:11
# !@author: superMC @email: 18758266469@163.com
# !@fileName: module.py
from torch import nn


class SubModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SubModule, self).__init__()

        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.fc2 = nn.Linear(in_channels // 2, in_channels // 8)
        self.fc3 = nn.Linear(in_channels // 8, in_channels // 32)
        self.fc4 = nn.Linear(in_channels // 32, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x
