"""
对图像生成空的txt文件
"""
import os

# 定义图像文件夹路径
image_folder = '/mnt/d/exp/qiyuan/negative_img/crop/'  # 替换为你的图像文件夹路径
txt_folder = '/mnt/d/exp/qiyuan/negative_img/crop_labels/'

os.makedirs(txt_folder, exist_ok=True)

# 遍历图像文件夹中的每一个文件
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg'):
        # 获取图像文件的基本名称（去掉扩展名）
        base_name = os.path.splitext(filename)[0]
        # 创建一个对应的空txt文件
        txt_path = os.path.join(txt_folder, f"{base_name}.txt")
        open(txt_path, 'w').close()  # 创建空文件

print("已为每个图像生成对应的空txt文件。")
