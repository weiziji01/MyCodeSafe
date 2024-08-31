# 将Isprs比赛数据集划分为训练集和测试集
import os
import numpy as np
import shutil
import tqdm

def write_txt(dataset, old_path, new_path, new_txt):
    """
    划分数据集并生成相应的标签
    dataset: 训练集/测试集数据对
    old_path: 原有图像存储路径
    new_path: 划分后图像存储路径
    new_txt: 划分后标签存储路径
    """
    for i in tqdm.trange(len(dataset)):
        img = old_path+dataset[i][0]
        shutil.copy(img, new_path)
        with open(new_txt, 'a') as f:
            f.write(dataset[i][0]+' '+dataset[i][1]+'\n')
    return 0

if __name__ == '__main__':
    txt_path = r'/mnt/d/data/isprs_game/train_set/gt.txt'   # 原有标签路径
    img_path = r'/mnt/d/data/isprs_game/train_set/image/'   # 原有数据集路径

    imgs = []
    with open(txt_path, 'r') as f:  # 读取标签，形成列表，内容是数据对(图片名，标签)
        for line in f:
            line = line.strip('\n')
            labels = line.split()
            imgs.append((labels[0], labels[1]))

    np.random.seed(seed=7)
    np.random.shuffle(imgs) # 打乱所有图像

    train_ratio = 0.8   # 训练集与测试集比例为8：2
    train_dataset = imgs[:int(len(imgs)*train_ratio)]
    test_dataset = imgs[int(len(imgs)*train_ratio):]

    train_path = r'/mnt/d/data/isprs_game/train_set/train/image/'   # 划分后训练集图像路径
    train_txt = r'/mnt/d/data/isprs_game/train_set/train/gt.txt'    # 划分后训练集标签路径
    test_path = r'/mnt/d/data/isprs_game/train_set/test/image/' # 划分后测试集图像路径
    test_txt = r'/mnt/d/data/isprs_game/train_set/test/gt.txt'  # 划分后测试集标签路径

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    print("训练集划分开始")
    write_txt(train_dataset, img_path, train_path, train_txt)
    print("训练集已划分完成")

    print("测试集划分开始")
    write_txt(test_dataset, img_path, test_path, test_txt)
    print("测试集已划分完成")
