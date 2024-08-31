# 比赛用的模型的代码
import torch
import torchvision
import torch.nn as nn
from torchvision import models
from torch.hub import load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class IsprsGameModel(models.ResNet):
    def __init__(self, num_classes=2, pre_trained=False, **kwargs):

        # 由resnet101预先定义的结构开始
        super().__init__(block=models.resnet.Bottleneck,layers=[3,4,23,3], num_classes=num_classes, **kwargs)
        if pre_trained:
            state_dict = load_state_dict_from_url(url=model_urls['resnet101'], progress=True)
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')
            self.load_state_dict(state_dict=state_dict, strict=False)
        
        # 将自适应池化代替标准池化方案
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        # 将resnet最后一层全连接层转换为卷积层
        self.last_conv = nn.Conv2d(in_channels=self.fc.in_features, out_channels=num_classes, kernel_size=1)
        self.last_conv.weight.data.copy_(self.fc.weight.data.view(*self.fc.weight.data.shape, 1, 1))
        self.last_conv.bias.data.copy_(self.fc.bias.data)

        # 将最后一层改成全连接层
        self.last_fc1 = nn.Linear(2048, 1024)
        self.last_fc2 = nn.Linear(1024, 512)
        self.last_fc3 = nn.Linear(512, 64)
        self.last_fc4 = nn.Linear(64, 2)
        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # resnet101的过程
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out4 = self.avgpool(out4)
        out4 = out4.view(-1,2048)
        out4 = self.last_fc1(out4)
        out4 = self.relu(out4)
        out4 = self.dropout(out4)
        out4 = self.last_fc2(out4)
        out4 = self.relu(out4)
        out4 = self.dropout(out4)
        out4 = self.last_fc3(out4)
        out4 = self.relu(out4)
        out4 = self.dropout(out4)
        out4 = self.last_fc4(out4)
        out4 = self.softmax(out4)

        # out4 = self.last_conv(out4)
        # out4 = out.view(-1, 8)
        # out4 = self.last_fc(out4)
        return out4
        
# if __name__ == '__main__':
#     print("ok")
#     a = torch.rand((4,3,200,200))
#     out = IsprsGameModel().forward(a)
#     # print(out)
#     print(out)
