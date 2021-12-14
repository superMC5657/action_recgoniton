# -*- coding: utf-8 -*-
# !@time: 2021/12/15 上午3:22
# !@author: superMC @email: 18758266469@163.com
# !@fileName: actionRecNet.py
from torch import nn

from .resnet import resnet18
from .module import SubModule


class ActionRecNet(nn.Module):
    def __init__(self):
        super(ActionRecNet, self).__init__()
        self.backbone = resnet18(pretrained=False, num_classes=1024)
        self.person_module = SubModule(1024, 2)
        self.sex_module = SubModule(1024, 2)
        self.name_module = SubModule(1024, 5)
        self.action_module = SubModule(1024, 6)

    def forward(self, x):
        x = self.backbone(x)
        person = self.person_module(x)
        sex = self.sex_module(x)
        name = self.name_module(x)
        action = self.action_module(x)
        return person, sex, name, action
