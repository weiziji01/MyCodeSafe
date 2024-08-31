"""
Pattern Recognition And Machine Learning
Convolutional Neural Network -- CIFAR-10
Created by Weiziji on 2023/11/29
"""
"""
待解决问题：
1. torch代码总结完善
2. 如何调用D盘的图像资料（没有在Linux系统所在的那个盘） -> 已解决，通过/mnt/实现
3. 算法还没有办法运行（在交叉熵损失那一行，可能得打印每个输出与标签的维度确认） -> 已解决，确实是维度错误，调试后解决
4. 补全注释
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# define VGG Network
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1=nn.Conv2d(3,64,kernel_size=3,padding=1)  # in_c=3; out_c=64
        self.conv2=nn.Conv2d(64,128,kernel_size=3,padding=1)  # in_c=64; out_c=128
        self.conv3=nn.Conv2d(128,256,kernel_size=3,padding=1)  # in_c=128; out_c=256
        self.conv4=nn.Conv2d(256,512,kernel_size=3,padding=1)  # in_c=256; out_c=512
        self.conv5=nn.Conv2d(512,512,kernel_size=3,padding=1)  # in_c=512; out_c=512
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(512,256)
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,10)
        self.relu=nn.ReLU(inplace=True)
        self.dropout=nn.Dropout()

    def forward(self,x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.pool(x)
        x=self.conv2(x)
        x=self.relu(x)
        x=self.pool(x)
        x=self.conv3(x)
        x=self.relu(x)
        x=self.pool(x)
        x=self.conv4(x)
        x=self.relu(x)
        x=self.pool(x)
        x=self.conv5(x)
        x=self.relu(x)
        x=self.pool(x)
        x=x.view(-1, 512)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.fc2(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.fc3(x)
        return x
        
# data pre-processing
transform_train=transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) # 是否与图像有关？？

transform_test=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Dataset
trainset=datasets.CIFAR10(root='/mnt/d/data/cifar10',train=True, download=True, transform=transform_train)
trainloader=torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset=datasets.CIFAR10(root='/mnt/d/data/cifar10',train=False, download=True, transform=transform_test)
testloader=torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Define model, loss function and optimizer
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net=VGG().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# Train the model
for epoch in range(50):
    running_loss=0.0
    for i, data in enumerate(trainloader,0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs=net(inputs)
        loss=criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %(epoch+1, i+1, running_loss/100))
            running_loss=0.0

    # Test the model
    correct=0
    total=0
    with torch.no_grad():
        for data in testloader:
            inputs, labels=data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs=net(inputs)
            _, predicted=torch.max(outputs.data, 1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    print("Accuracy of the network on the 10000 test images: %d %%" %(100*correct/total))

