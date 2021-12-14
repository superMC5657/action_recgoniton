# -*- coding: utf-8 -*-
# !@time: 2021/12/15 上午2:33
# !@author: superMC @email: 18758266469@163.com
# !@fileName: config.py
import os

train_threshold = 0.8
val_threshold = 0.1

# 需要修改自己的路径
input_dir = "/home/supermc/Downloads/fallDetect"

train_csv_path = os.path.join(input_dir, "train.csv")
val_csv_path = os.path.join(input_dir, "val.csv")
test_csv_path = os.path.join(input_dir, "test.csv")

sex_label_dict = {"a": 0, "b": 1, "c": 0, "d": 1, "e": 1}
name_label_dict = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
person_label_dict = {"6": 0, "1": 1, "2": 1, "3": 1, "4": 1, "5": 1}
action_label_dict = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5}
