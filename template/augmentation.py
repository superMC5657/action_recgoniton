# -*- coding: utf-8 -*-
# !@time: 2021/12/14 上午2:06
# !@author: superMC @email: 18758266469@163.com
# !@fileName: augmentation.py


# 透视变换
import random

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from multiTask import data_dir


def random_warp(img, row, col):
    height, width, channels = img.shape
    # warp:
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return M_warp, img_warp


# 改变颜色
def random_light_color(img):
    # brightness
    B, G, R = cv2.split(img)  # 通道拆分，顺序为BGR,不是RBG

    b_rand = random.randint(-50, 50)  # 生成随机数整数n a<=n<=b
    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B > lim] = 255
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < lim] = 0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)

    g_rand = random.randint(-50, 50)
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255  # R[],G[],B[]都是矩阵
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)

    r_rand = random.randint(-50, 50)
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
    elif r_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)

    img_merge = cv2.merge((B, G, R))  # 合并之前分离出来进行变换的通道
    # img = cv2.cvtColor(final_hsv,cv2.COLOR_HSV2BGR)
    return img_merge


# 对图片实现多种变换并保存
def image_data_aug(img, crop=True, change_color=True, rotation=True, perspective_transform=False):
    if (crop or change_color or rotation or perspective_transform) == False:
        print("wrong input")
        return
    if crop:
        img = img[int(img.shape[0] / 4):int(3 * img.shape[0] / 4), 0:int(3 * img.shape[1] / 4)]  # 根据图像大小选择参数大小
    if change_color:
        img = random_light_color(img)
    if rotation:
        angle = random.randint(0, 180)
        scale = random.uniform(0.75, 1.25)
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, scale)  # center, angle, scale
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    if perspective_transform:
        M_warp, img = random_warp(img, img[0], img[1])

    return img


def create_image(ori_img_file, times=1):
    """
    param:  ori_img_file:记录训练数据集相关信息的csv文件路径+名称；
            times:      为数据集增加的倍数
    """
    ori_file = pd.read_csv(data_dir + ori_img_file, index_col=0)
    new_csv = []
    for time in range(times):
        print("图片第{}次生成中...".format(str(time)))
        with tqdm(range(len(ori_file))) as t:
            for idx in t:
                ori_path = ori_file["path"][idx]
                path = ori_path.replace(".jpg", "_aug" + str(time) + "_" + str(idx) + ".jpg").replace("train",
                                                                                                      "train_aug").replace(
                    "val", "val_aug")
                classes = ori_file["classes"][idx]
                species = ori_file["species"][idx]
                print(ori_path)
                img = cv2.imread(ori_path)

                try:
                    img = image_data_aug(img)
                    # """
                    # 不知道为啥遍历到idx=680时，总会显示error: C:\projects\opencv-python\opencv\modules\highgui\src\window.
                    # cpp:325: error: (-215) size.width>0 && size.height>0 in function cv::imshow
                    # (已经改斜杠，确认路径没有中文，所以用了try...except这个结构)
                    cv2.imwrite(path, img)
                    new_csv.append([str(time) + "_" + str(idx), path, classes, species])

                except:
                    continue
    data_aug = pd.DataFrame(new_csv, columns=["index", "path", "classes", "species"])
    data_aug.to_csv("data_aug.csv", index=0)
