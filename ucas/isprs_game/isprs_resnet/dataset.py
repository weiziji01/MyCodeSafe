# 加载比赛用的数据集的代码
import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

# data_path = r'/mnt/d/data/isprs_game/train_set/image'   # 图像存放地址
# txt_path = r'/mnt/d/data/isprs_game/train_set/gt.txt'   # 标签存放地址

# transform = transforms.Compose([
#     transforms.Resize(400),
#     transforms.RandomCrop((200,200)),
#     transforms.ToTensor()
# ])

class IsprsGameDataset(Dataset):
    def __init__(self, txt_path, data_path, transform=None):
        imgs=[] # 存储标签与图片信息

        # data_info = open(txt_path, 'r')
        with open(txt_path, 'r') as data_info:
            for line in data_info:  # 打开标签文档，按行读取
                line = line.strip('\n')
                labels = line.split()
                imgs.append((labels[0], labels[1])) # 以元组的形式存储，图片名与标签信息

        self.imgs = imgs
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.imgs)

    def __getitem__(self, index):
        """
        打开对应index的图像，进行预处理后，返回该图像与标签
        """
        data_path = self.data_path
        img, label = self.imgs[index]
        img_path = data_path + '/' +img
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        label=np.array(label).astype(int)
        label=torch.from_numpy(label)
        return img, label
