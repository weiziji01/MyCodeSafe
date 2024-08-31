# 调用torchvisio中的resnet，完成真伪图像鉴别
import time
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataset import IsprsGameDataset
from model import IsprsGameModel

transforms_train = transforms.Compose([
    transforms.Resize(400),
    transforms.RandomCrop((200,200)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
transforms_test = transforms.Compose([
    transforms.Resize(400),
    transforms.RandomCrop((200,200)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# 训练超参数
Download=True
EPOCHS=10
LR=0.001
BATCHSIZE=64

data_path = r'/mnt/d/data/isprs_game/train_set/image'   # 图像存放地址
txt_path = r'/mnt/d/data/isprs_game/train_set/gt.txt'   # 标签存放地址

train_data = IsprsGameDataset(txt_path, data_path, transform=transforms_train)
test_data = IsprsGameDataset(txt_path, data_path, transform=transforms_test)

train_size = len(train_data)
test_size = len(test_data)

train_dataloader = DataLoader(dataset=train_data, batch_size=BATCHSIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCHSIZE, shuffle=True)

# 调用已训练好的ResNet
model = IsprsGameModel(pre_trained=True)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet101 = model.to(device)
optimizer = torch.optim.Adam(resnet101.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# 开始训练
train_loss = []
train_acc = []
test_loss = []
test_acc = []
for epoch in range(EPOCHS):
    running_loss = 0.0
    start_time = time.perf_counter()
    print("第{}轮训练开始：".format(epoch+1))
    total_train_loss = 0
    total_train_acc = 0
    total_test_loss = 0
    total_test_acc = 0

    # 训练部分
    for step, data in enumerate(train_dataloader):
        img, ann = data
        img = img.to(device)
        ann = ann.to(device)
        out = resnet101(img)
        loss = loss_func(out, ann)
        total_train_loss += loss.data.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_pred_y = torch.max(out, 1)[1].data.cpu().numpy()
        train_accuracy = (train_pred_y == ann.data.cpu().numpy()).astype(int).sum()
        total_train_acc += train_accuracy.item()

        running_loss += loss.item()
        if step % 32 == 31:
            print('[%d, %5d] loss: %.3f' %(epoch+1, step+1, running_loss/100))
            running_loss=0.0
    
    train_loss.append(total_train_loss/train_size)
    train_acc.append(100*total_train_acc/train_size)
    print("训练集上的loss：{}， 正确率：{:.2f}%".format(train_loss[epoch], train_acc[epoch]))

    # 测试部分
    with torch.no_grad():
        for step, data in enumerate(test_dataloader):
            img, ann = data
            img, ann = img.to(device), ann.to(device)
            out = resnet101(img)
            pred_y = torch.max(out, 1)[1].data.cpu().numpy()
            loss = loss_func(out, ann)
            total_test_loss += loss.data.item()
            accuracy = (pred_y == ann.data.cpu().numpy()).astype(int).sum()
            total_test_acc += accuracy.item()

    test_loss.append(total_test_loss/test_size)
    test_acc.append(100*total_test_acc/test_size)
    print("测试集上的loss：{}，正确率：{:.2f}%".format(test_loss[epoch], test_acc[epoch]))

    end_time = time.perf_counter()
    print("Epoch: {}, 用时: {}".format(epoch, end_time-start_time))
    

