"""
Pattern Recognition And Machine Learning
Neural Network -- MNIST
Created by Weiziji on 2024/01/08
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsummary import summary
import time

# 导入数据集
train_dataset=datasets.MNIST(root='/mnt/d/data/mnist',train=True,transform=transforms.ToTensor(),download=True)
test_dataset=datasets.MNIST(root='/mnt/d/data/mnist',train=False,transform=transforms.ToTensor(),download=True)

# 数据加载
train_loader=DataLoader(dataset=train_dataset, batch_size=100,shuffle=True)
test_loader=DataLoader(dataset=test_dataset, batch_size=100,shuffle=True)

# 网络定义
class Model_MNIST(torch.nn.Module):
    def __init__(self):
        super(Model_MNIST, self).__init__()
        self.conv1=torch.nn.Sequential(torch.nn.Conv2d(1,64,3,1,1), # in_c=1; out_c=64
                                       torch.nn.ReLU(), # conv1中图像尺寸仍为28*28
                                       torch.nn.Conv2d(64,128,3,1,1),
                                       torch.nn.ReLU(),
                                       torch.nn.MaxPool2d(2,2))
        self.dense=torch.nn.Sequential(torch.nn.Linear(14*14*128,1024),# 此时图像尺寸14*14
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(1024,10))
        
    def forward(self,x):
        x=self.conv1(x)
        x=x.view(-1,14*14*128)  # flatten操作
        x=self.dense(x)
        return x
    
# 定义模型、损失函数、优化方法
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=Model_MNIST().to(device)
cost=torch.nn.CrossEntropyLoss()    # 交叉熵损失
optimizer=torch.optim.Adam(model.parameters())

# 主程序
if __name__ == "__main__":
    start=time.perf_counter()
    # 训练过程
    for epoch in range(5):
        sum_loss=0.0
        train_correct=0
        for data in train_loader:
            inputs,labels=data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs=model(inputs)
            loss=cost(outputs,labels)
            loss.backward() # 后向计算
            optimizer.step()    # 更新所有参数

            _, pred=torch.max(outputs.data,1)   # 返回最大值与其索引，pred即是其索引
            sum_loss+=loss.data
            train_correct+=torch.sum(pred==labels.data) # 判断正确时便加1

        print('Epoch[%d/%d] loss: %.03f' %(epoch+1,5,sum_loss/len(train_loader)))
        print('        correct: %.03f%%' %(100*train_correct/len(train_dataset)))
    
    # 测试过程
    model.eval()    # 将模型设置为测试模式
    test_correct=0
    with torch.no_grad():   # 关闭梯度计算
        for data in test_loader:
            inputs,labels=data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs=model(inputs)
            _, pred=torch.max(outputs.data,1)
            test_correct+=torch.sum(pred==labels.data)
        print("correct: %.03f%%" %(100*test_correct/len(test_dataset)))

    summary(model=model, input_size=(1,28,28), batch_size=100)

    # 计算耗时
    end=time.perf_counter()
    print("Totally time: ", end-start)
