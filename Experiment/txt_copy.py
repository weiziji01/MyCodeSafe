"""
文件夹内文件的转移
"""

import os
import shutil

# 设置甲文件夹、乙文件夹和丙文件夹的路径
folder_a = "/mnt/d/exp/dji_car/datasets_have_error_split/train/labels"
folder_b = "/mnt/d/data/dji_data/labelTxt_dota"
folder_c = "/mnt/d/exp/dji_car/datasets_dota/train/labelTxt/"

# 创建丙文件夹（如果不存在）
if not os.path.exists(folder_c):
    os.makedirs(folder_c)

# 获取甲文件夹中的所有文件
files_in_a = set(os.listdir(folder_a))

# 遍历乙文件夹中的文件
for root, dirs, files in os.walk(folder_b):
    for file in files:
        # 如果乙文件夹中的文件名与甲文件夹中的文件名一致，则复制该文件
        if file in files_in_a:
            # 获取源文件路径
            source_file = os.path.join(root, file)
            # 获取目标文件路径
            destination_file = os.path.join(folder_c, file)
            # 复制文件到丙文件夹
            shutil.copy(source_file, destination_file)
            print(f"已复制: {file}")
