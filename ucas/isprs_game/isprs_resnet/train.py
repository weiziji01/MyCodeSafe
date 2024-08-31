# 调用torchvisio中的resnet，完成真伪图像鉴别
import time
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataset import IsprsGameDataset
from model import IsprsGameModel
from losses import weighted_FocalLoss

def train_dataset(train_dataloader, epoch, device, model, loss_func, optimizer):
    """
    对数据集进行训练
    """
    running_loss = 0.0
    epoch_train_loss = 0

    # 训练部分
    for i, data in enumerate(train_dataloader):
        img, ann = data
        img = img.to(device)
        ann = ann.to(device)
        out = model(img)

        loss = loss_func(out, ann.long())
        if torch.isnan(loss):
            print('---nan---')
            
        epoch_train_loss += loss.data.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印指定batch的损失
        running_loss += loss.item()
        if i % 32 == 31:
            print('Epoch:{}, Batch:{} | loss: {}'.format(epoch+1, i+1, running_loss))
            running_loss=0.0

    return 0

def test_dataset(test_dataloader, device, model, epoch):
    """
    对训练好的模型进行测试
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            img, ann = data
            img = img.to(device)
            ann = ann.to(device)
            out = model(img)
            _, predicted = torch.max(out.data, 1)
            total += ann.size(0)
            correct += (predicted == ann).sum().item()
    print("EPOCH {} test Accuracy: {}".format(epoch+1, 100*correct/total))

    return 0 


if __name__ == '__main__':
    # 训练参数设置
    transforms_train = transforms.Compose([
        # transforms.Resize(400),
        transforms.RandomCrop((200,200)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    transforms_test = transforms.Compose([
        # transforms.Resize(400),
        transforms.RandomCrop((200,200)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    # 训练超参数
    EPOCHS=10
    LR=0.001
    BATCHSIZE=32

    # data_path = r'/mnt/d/data/isprs_game/train_set/image'   # 图像存放地址
    # txt_path = r'/mnt/d/data/isprs_game/train_set/gt.txt'   # 标签存放地址

    train_path = r'/mnt/d/data/isprs_game/train_set/train/image'    # 训练集图像路径
    train_txt = r'/mnt/d/data/isprs_game/train_set/train/gt.txt'    # 测试集图像路径
    train_data = IsprsGameDataset(train_txt, train_path, transform=transforms_train)
    train_size = len(train_data)
    train_dataloader = DataLoader(dataset=train_data, batch_size=BATCHSIZE, shuffle=True)

    test_path = r'/mnt/d/data/isprs_game/train_set/test/image'  # 测试集图像路径
    test_txt = r'/mnt/d/data/isprs_game/train_set/test/gt.txt'  # 测试集标签路径
    test_data = IsprsGameDataset(test_txt, test_path, transform=transforms_test)
    test_size = len(test_data)
    test_dataloader = DataLoader(dataset=test_data, batch_size=BATCHSIZE, shuffle=True)

    # 调用已训练好的ResNet
    # [batch h w c]->[batch c h w]..reshape->view[batch,H*W,c]->[batch c*H*W]->lin{xxxxxxx}->[batch 2]
    model = IsprsGameModel(pre_trained=True)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet101 = model.to(device)
    optimizer = torch.optim.Adam(resnet101.parameters(), lr=LR)

    binary_lossFc = weighted_FocalLoss(num_class=2)
    binary_lossFc = binary_lossFc.cuda()

    # 开始训练
    for epoch in range(EPOCHS):
        start_time = time.perf_counter()
        print("----------EPOCH: {}-----------".format(epoch+1))
        train_dataset(train_dataloader, epoch, device, resnet101, binary_lossFc, optimizer)
        test_dataset(test_dataloader, device, resnet101, epoch)
        end_time = time.perf_counter()
        print("EPOCH {} time: {}".format(epoch+1, end_time-start_time))
