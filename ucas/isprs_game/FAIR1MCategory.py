# FAIR1M数据集的类别统计
import os
import os.path
import numpy as np
import cv2
import matplotlib.pyplot as plt

train_list = os.listdir("/mnt/d/data/fair1m1/train/part1/labelTxt/")
base_path = "/mnt/d/data/fair1m1/train/part1/labelTxt/"

total_dict = []

for file in train_list:
    with open(base_path + file) as f:
        s = f.readlines()
        for si in s:
            bbox_info = si.split()
            if bbox_info[8] in total_dict:
                continue
            else:
                total_dict.append(bbox_info[8])
print(total_dict)
print(len(total_dict))
