# -*- coding: utf-8 -*-
# !@time: 2021/12/14 上午1:51
# !@author: superMC @email: 18758266469@163.com
# !@fileName: generate_data.py

import os
import csv
import random

from config import action_label_dict, person_label_dict, sex_label_dict, name_label_dict, val_threshold, \
    train_threshold, input_dir, test_csv_path, val_csv_path, train_csv_path

train_csv = open(train_csv_path, "w")
val_csv = open(val_csv_path, "w")
test_csv = open(test_csv_path, "w")

train_csv_writer = csv.writer(train_csv)
val_csv_writer = csv.writer(val_csv)
test_csv_writer = csv.writer(test_csv)

train_csv_writer.writerow(
    ["rgb_image_path", "depth_image_path", "person_label", "sex_label", "name_label", "action_label"])
val_csv_writer.writerow(
    ["rgb_image_path", "depth_image_path", "person_label", "sex_label", "name_label", "action_label"])
test_csv_writer.writerow(
    ["rgb_image_path", "depth_image_path", "person_label", "sex_label", "name_label", "action_label"])


# 获取一个文件夹中的所有子文件夹
def get_sub_dirs(root_dir):
    sub_dirs = os.listdir(root_dir)
    sub_dirs = [os.path.join(root_dir, sub_dir) for sub_dir in sub_dirs]
    sub_dirs = [sub_dir for sub_dir in sub_dirs if os.path.isdir(sub_dir)]
    return sub_dirs


sub_dirs = get_sub_dirs(root_dir=input_dir)

for sub_dir in sub_dirs:
    rgb_dir = os.path.join(sub_dir, "rgb")
    depth_dir = os.path.join(sub_dir, "depth")
    label_path = os.path.join(sub_dir, "labels.csv")
    label_file = open(label_path, "r")
    label_reader = sorted(list(csv.reader(label_file))[1:], key=lambda x: int(x[0]))
    name_label = name_label_dict[sub_dir[-1]]
    sex_label = sex_label_dict[sub_dir[-1]]
    for index, label in enumerate(label_reader):
        rgb_image_path = os.path.join(rgb_dir, "rgb_" + ("0000" + label[0])[-4:] + ".png")
        depth_image_path = os.path.join(depth_dir, "depth_" + ("0000" + label[0])[-4:] + ".png")
        if label[1] not in action_label_dict:
            continue
        if not os.path.exists(rgb_image_path) or not os.path.exists(depth_image_path):
            continue
        person_label = person_label_dict[label[1]]
        action_label = action_label_dict[label[1]]
        random_num = random.random()
        if random_num < train_threshold:
            train_csv_writer.writerow(
                [rgb_image_path, depth_image_path, person_label, sex_label, name_label, action_label])
        elif random_num < train_threshold + val_threshold:
            val_csv_writer.writerow(
                [rgb_image_path, depth_image_path, person_label, sex_label, name_label, action_label])
        else:
            test_csv_writer.writerow(
                [rgb_image_path, depth_image_path, person_label, sex_label, name_label, action_label])
