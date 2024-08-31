# Resnet模型构建代码
import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class BasicBlock(nn.Module):
    """
    ResNet18/34, 两个3*3卷积
    """
    expansion=1 # 残差结构中，主分支卷积核数目是否变化，不变则为1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        # 3*3卷积
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, 
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        # 3*3卷积
        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, 
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        residual = x    # 残差
        if self.downsample is not None: # 支路，不进行下采样的话，与原来保持一致
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    """
    ResNet50/101/152, 1*1+3*3+1*1卷积
    """
    expansion=4 # 残差结构中第3层卷积核个数是第1/2层卷积核个数的4倍
                # 经过该块后，通道数变为原来的4倍，64至256

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1*1 卷积
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, 
                             kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        # 3*3 卷积
        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, 
                             kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        # 1*1的4倍的卷积
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*4, 
                             kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x    # 残差
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """
    ResNet model的搭建
    """
    def __init__(self, num_classes, block, layers):
        self.in_channel = 64
        super(ResNet, self).__init__()
        # 7*7卷积
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 3*3池化
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet的四层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_channel, blocks, stride=1):
        """
        # 实现ResNet的每一层，共四层，每层里残差块的数量在ResNet50或101中是不同的
        """
        downsample = None
        if stride != 1 or self.in_channel != out_channel * block.expansion: # 使得残差与后面输出维度相同的措施
            downsample = nn.Sequential( # 每个残差块的支路的生成
                nn.Conv2d(in_channels=self.in_channel, out_channels=out_channel*block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel*block.expansion),
            )
        
        layers = [block(self.in_channel, out_channel, stride, downsample)]
        self.in_channel = out_channel * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channel, out_channel))  # 四层中，小部分层进行整合

        return nn.Sequential(*layers)
    
    def forward(self, inputs):
        img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x2, x3, x4
    
def resnet18(num_classes, pre_trained=False, **kwargs):
    """
    构建的是ResNet18模型
    """
    model = ResNet(num_classes, BasicBlock, [2,2,2,2], **kwargs)
    if pre_trained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model

def resnet34(num_classes, pre_trained=False, **kwargs):
    """
    构建的是ResNet34模型
    """
    model = ResNet(num_classes, BasicBlock, [3,4,6,3], **kwargs)
    if pre_trained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model

def resnet50(num_classes, pre_trained=False, **kwargs):
    """
    构建的是ResNet50模型
    """
    model = ResNet(num_classes, Bottleneck, [3,4,6,3], **kwargs)
    if pre_trained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model

def resnet101(num_classes, pre_trained=False, **kwargs):
    """
    构建的是ResNet101模型
    """
    model = ResNet(num_classes, Bottleneck, [3,4,23,3], **kwargs)
    if pre_trained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model

def resnet152(num_classes, pre_trained=False, **kwargs):
    """
    构建的是ResNet152模型
    """
    model = ResNet(num_classes, Bottleneck, [3,8,36,3], **kwargs)
    if pre_trained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model
