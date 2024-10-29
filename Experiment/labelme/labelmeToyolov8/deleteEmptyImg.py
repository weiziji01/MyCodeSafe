# 删除没有标签的图片
# 标签与图片需要在不同的文件夹中

from PIL import Image
import os
import os.path
import numpy as np
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import time

start = time.time()
n = 0
# 指明被遍历的文件夹
# ! 下面是路径修改的地方
rootdir = 'datasets/images/'#图片路径
json_path = 'datasets/label_json/'  #标签文件存放的文件夹地址
# ! 上面是路径修改的地方
for parent, dirnames, filenames in os.walk(rootdir):
    # 遍历每一张图片
    for filename in filenames:
        currentPath = os.path.join(parent, filename)  # 目前图片的地址
        picture_number = filename[:-4]  # 图片去掉.png（.jpg）后的名字
        print("the picture name is:" + filename)
        json_file = os.listdir(json_path)  # 将标签文件存放在json_file列表里面
        if picture_number + '.json' in json_file:  # 如果图片有对应的标签文件则跳过
            pass
        else:
            print(currentPath)
            n += 1
            os.remove(currentPath)  # 没有对应的标签文件则删除
            end = time.time()
end = time.time()
print("Execution Time:", end - start)
print("删除的照片数为：", n)
