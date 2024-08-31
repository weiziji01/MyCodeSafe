"""
Pattern Recognition And Machine Learning
Logistic Regression-- MNIST
Created by Weiziji on 2024/01/08
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# 导入数据集
train_dataset=datasets.MNIST(root='/mnt/d/data/mnist',train=True,transform=transforms.ToTensor(),download=True)
test_dataset=datasets.MNIST(root='/mnt/d/data/mnist',train=False,transform=transforms.ToTensor(),download=True)

# 数据加载
train_loader=DataLoader(dataset=train_dataset, batch_size=100,shuffle=True)
test_loader=DataLoader(dataset=test_dataset, batch_size=100,shuffle=True)
    
# 定义模型、损失函数、优化方法
model=torch.nn.Linear(28*28, 10)    # 逻辑回归模型在torch中即可直接使用Linear函数表达
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
            inputs=inputs.reshape(-1,28*28)
            optimizer.zero_grad()
            outputs=model(inputs)
            loss=cost(outputs,labels)
            loss.backward()
            optimizer.step()

            _, pred=torch.max(outputs.data,1)   
            sum_loss+=loss.data
            train_correct+=torch.sum(pred==labels.data)

        print('Epoch[%d/%d] loss: %.03f' %(epoch+1,5,sum_loss/len(train_loader)))
        print('        correct: %.03f%%' %(100*train_correct/len(train_dataset)))
    
    # 测试过程
    model.eval()
    test_correct=0
    with torch.no_grad():
        for data in test_loader:
            inputs,labels=data
            inputs=inputs.reshape(-1,28*28)
            outputs=model(inputs)
            _, pred=torch.max(outputs.data,1)
            test_correct+=torch.sum(pred==labels.data)
        print("correct: %.03f%%" %(100*test_correct/len(test_dataset)))
    
    # 计算耗时
    end=time.perf_counter()
    print("Totally time: ", end-start)