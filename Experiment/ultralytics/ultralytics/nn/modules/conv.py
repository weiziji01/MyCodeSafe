# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs.返回pad的大小，使得padding后输出张量的大小不变
        k: (kernel) int or 序列，卷积核的大小
        p: (padding) None, 填充的大小
        d: (dilation rate), 默认为1, 扩张率的大小。普通卷积的扩张率为1, 空洞卷积的扩张率大于1    
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
        # 加入空洞之后的实际卷积核尺寸与原始卷积核尺寸之间的关系: k=d(k-1)+1
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
        # //是向下整除运算，类似于math.floor()。示例即29//10=2
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation).
        c1: 输入通道数; c2: 输出通道数(卷积核的数量就是c2); k: 卷积核的大小;
        s: 步长，默认为1; p: 填充，默认为None; g: 组，默认为1;
        d: 扩张率，默认为1; act: 是否采用激活函数，默认为True，且采用SiLU为激活函数;
        SiLU(x)=x(1/(1+exp(-x)))
        对于group参数，假如group=2，等效于并排两个卷积层，每个层输入1/2c1并输出1/2c2，并且随后将二者连起来
        空洞卷积层与一般卷积间的差别在于膨胀率，膨胀率控制的是卷积时的 padding 以及 dilation。
            通过不同的填充以及与膨胀，可以获取不同尺度的感受野，提取多尺度的信息。
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)    # 均值为0, 方差为1
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        # 如果act=True, 则采用默认的激活函数SiLU; 如果act的类型是nn.Module, 则采用传入的act;
        # 否则不采取任何动作(nn.Identity函数相当于f(x)=x, 只用做占位, 返回原始的输入)

    def forward(self, x):   # 标准的前向传播算法，通常在训练和推理时都使用
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):  # 用Module类的fuse函数融合Conv+BN加速推理, 一般用于测试/验证阶段
        # 这是一个优化的前向传播方法, 将BN批归一化层和conv卷积层融合, 通常用于推理阶段
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing.
        实现普通卷积和1*1卷积的叠加
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv
        # cv2设置的是1*1的卷积，可用于整合特征图不同通道之间的信息。只在通道维度上进行计算，从而混合各通道的信息。能用来升维或降维

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))
        # 使用两个卷积操作(1个普通卷积和1个1*1卷积),并将其结果相加
        # 增加模型复杂性和表达能力，用于训练阶段，充分利用模型的复杂性

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))
        # 仅使用一个卷积操作，适用于需要简化计算和加速推理的场景，用于推理阶段，减少计算量和提高推理速度

    def fuse_convs(self):
        """Fuse parallel convolutions.
            目的是将两个并行的卷积操作融合成一个，以简化模型结构并提高推理速度
            将self.cv2(1*1卷积)的卷积核权重整合到self.conv的权重中，并删除self.cv2层，使模型在推理阶段只执行一次卷积操作
        """
        w = torch.zeros_like(self.conv.weight.data) # w用于存储self.cv2的权重数据 
            # 卷积层的权重通常为思维张量，形状为(OutChannels, InChannels, KernelHeight, KernelWidth)
        i = [x // 2 for x in w.shape[2:]]   # 取出(KernelHeight, KernelWidth)卷积核的高度和宽度，并计算他们的一半(卷积核的中心)
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()    # 将self.cv2的权重数据放在w的中心位置
            # clone方法为了确保复制的是数据的副本 
        self.conv.weight.data += w  # 将嵌入了self.cv2的权重w加到了self.conv的权重上，从而将self.cv2的权重整合进self.conv
        self.__delattr__("cv2") # 删除了self.cv2属性，融合后不再需要self.cv2层
        self.forward = self.forward_fuse    # 更新forward方法，使其指向forward_fuse.使模型在推理阶段只执行一次卷积操作，不涉及1*1卷积


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    实现深度可分离卷积(Depthwise Separable Convolution)
    即在深度卷积之后，通过Pointwise Convolution操作生成新的Feature Map
    由深度卷积和逐点卷积(1*1卷积)组成，深度卷积用于提取空间特征，逐点卷积用于提取通道特征
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False) # 创建1*1卷积，即Pointwise Convolution(将输入在深度方向进行加权组合)
        self.conv2 = DWConv(c2, c2, k, act=act) # Depthwise Convolution

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))    # 依次应用1*1卷积和深度卷积提取特征
            # 减少计算量，同时保持一定的特征提取能力，适用于需要降低模型复杂度同时保持精度的场景


class DWConv(Conv):
    """Depth-wise convolution.
        深度卷积：一个卷积核负责一个通道，一个通道只被一个卷积核卷积
                 卷积核的数量与上一层的通道数相同（通道核卷积核一一对应）
                 无法扩展Feature Map，对输入层的每个通道独立进行卷积运算，没有有效利用不同通道在相同空间位置的feature信息
                 通常需要Pointwise Convolution将这些Feature Map生成新的Feature Map
    """

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
        # math.gcd计算c2(输出通道)与c1(输入通道)的最大公约数--深度卷积的关键
        # 设置为最大公约数，确保了组数能够同时整除输入通道和输出通道，从而允许每个滤波器处理单个输入通道，同时仍然生成所需的输出通道


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution.
        该类只是把"深度"操作Depthwise用在转置卷积上
    """

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters.
            p1输入填充，在转置卷积操作之前应用于输入特征图，可以控制输入特征图边界如何处理(零填充或镜像填充)
                以及填充的宽度，会影响输出特征图的大小。
            p2输出填充，在转置卷积操作之后应用于输出特征图，不会影响输出特征图的通道数，而是会改变其高和宽维度的大小
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))
        # 定义了groups为c1和c2的最小公约数，实现深度转置卷积的关键
        # 使得每一个转置卷积过滤器都只应用于一层输入层，可减小转置卷积所需的参数


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer.
        转置卷积，适用于需要进行上采样操作或重建丢失信息的场景。
        转置卷积又称反卷积(Deconvolution)，其上采样方式并非预设的插值方法，而是具有可学习的参数，可通过网络学习获取最优的上采样方式
        常规卷积的操作是不可逆的，所以转置卷积并不是通过输出矩阵和卷积核计算原始输入矩阵，而是计算得到保持了相对位置关系的矩阵
        矩阵中的实际权值不一定来自原始卷积矩阵，但权重的排布由卷积矩阵的转置的来。转置卷积与普通卷积形成相同的连通性但方向相反
        转置卷积不是卷积，但可以用卷积来模拟转置卷积。通过在输入矩阵的值间插入零值(以及周围填零)上采样输入矩阵，然后进行常规卷积，
            就会产生与转置卷积相同的效果
        注意：转置卷积会导致生成图像中出现的网格/棋盘效应(checkerboard artifacts)
                棋盘效应由于反卷积的"不均匀重叠(Uneven overlap)"的结果，使图像中某个部位的颜色比其他部位颜色更深
                    具体原因：反卷积操作时，卷积核的大小不能被步长整除，反卷积输出的结果就会不均匀重叠
                原则上，网络可以通过训练调整权重来避免此情况，调整好卷积核大小与步长之间的关系(不重叠与均匀重叠均可避免)
                    还可以进行插值Resize，再进行反卷积操作来避免
        (转置卷积的概念、定义、实现等从CSDN中观看更加直观)
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        # bias=not bn: 如果bn为False(不使用批归一化),则添加偏置项
        # 这是因为批归一化本身会添加一个偏置项，因此在这种情况下不需要额外的偏置项
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        # 如果bn参数是True，那么创建批归一化层；如果用户指定bn为False，那么使用空占位符nn.Identity()，创建f(x)=x
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):   # 训练与推理
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):  # 推理
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module): 
    """Focus wh information into c-space.
        主要功能是降低输入张量的空间尺寸(高度和宽度)，同时增加通道数。
        输入是(B,C,H,W),输出是(B,4C,H/2,W/2)
        降低空间分辨率，加快计算速度并降低内存使用量，尤其在处理大型图像时
        通过连接每个通道中不同空间位置的元素，模型可以潜在捕捉到图像中特征的更全面表示
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        变量中x的本来形状是(B,C,H,W)，送入卷积时特征图的形状为(B,4C,H/2,W/2)
        前向传播时对每一个通道切片(,1,H,W)都进行四次采样，并在通道维度上叠加，得到(,4,H/2,W/2)的切片，再把整体(B,4C,h/2,W/2)送入卷积
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))
        # x[..., ::2, ::2]是指取了前两个维度Batchsize和Channel的所有值，按照一定步长取了Height和Width的部分值
        # x[..., ::2, ::2]取出偶数行和偶数列的元素
        # x[..., 1::2, ::2]取出奇数行和偶数列的元素
        # x[..., ::2, 1::2]取出偶数行和奇数列的元素
        # x[..., 1::2, 1::2]取出奇数行和奇数列的元素
        # torch.cat将这四个子张量沿通道维度(轴1)进行拼接。有效地将来自所有四种采样方式的信息组合成一个单一张量，该张量通道数是原始通道数四倍


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet.
        GhostConv幻影卷积，旨在通过廉价操作生成更多的特征图。
        神经网络的临近层经常会生成一些类似的特征图，用普通卷积生成这些类似的特征图很耗费资源
        步骤：先进行1*1卷积聚合通道间的信息特征，然后再使用分组卷积，生成新的特征图
        为了减少网络计算量，作者将传统卷积分为两步及逆行，首先通过传统卷积生成channel较小的特征图以减少计算量，
            然后在得到的特征图的基础上，通过cheap operation(depthwise conv，廉价操作)再进一步减少计算量，生成新的特征图
            最后将两组特征图拼接到一起，生成最终的输出
        卷积操作是卷积-批归一化BN-非线性激活全套组合，而所谓的线性变换或廉价操作(cheap operation)均指普通卷积，不含批归一化和非线性激活
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels，输出的一半
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act) # 普通卷积
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)    # 深度卷积，c_为深度卷积的分组
        # cv2使用的卷积核5*5的深度卷积即是作者所言的"廉价的操作"

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)   # torch.cat()：在给定维度上对输入的张量序列seq进行连接操作
        # torch.cat((y,self.cv2(y)),1)意味着把y和self.cv2(y)沿着通道的方向(1的方向)连接
        # 相当于是把cv1(x)和cv2(cv1(x))沿着通道方向连接
        # cv1(x)的通道数是c_，cv2(cv1(x))的通道数还是c_，由于c_是c2的一半，两个通道相加之后就变成了c2，即输出通道数


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    RepConv：重参数化卷积。将传统的卷积层拆分为深度卷积和逐点卷积，即深度可分离卷积，并引入了一个新的参数q来控制两者之间的比例。
        通过实验发现，q=0.5时，模型可以获得最佳性能。
        这种结构重参数化方法可以有效减少模型的参数量和计算量，同时又不损失精度。
        RepConv在训练时还使用多分支结构，在推理时将多分支的参数融合，使用单分支结构。
            这种训练多分支、推理单分支的结构可以节省计算资源。
    
    RepConv类在训练的时候使用三个并列的分支(self.conv1是3x3卷积、self.conv2是1x1卷积、self.bn是BN层)，
    前向传播使用forward函数，在通过反向传播训练好三个分支的参数后，
    使用_fuse_bn_tensor函数、_pad_1x1_to_3x3_tensor函数、get_equivalent_kernel_bias函数、
    fuse_convs函数去融合三个分支，并获得融合后的单分支(一层卷积)，用于最后的推理过程
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):  # deploy是否处于部署状态(默认为False)
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        # assert断言关键字，如果后面的条件不满足(不为真)，那么会触发AssertionError异常
        # 即k=3,p=1这两件事必须同时发生，否则会引发异常
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        # 如果用户给的参数bn不为False而且输入通道数c1等于输出通道数c2而且卷积步幅s=1
        # 那么self.bn是一层用nn.BatchNorm2d实现的BN层，通道数为c1
        # 上述限定条件之一有一个不满足，则self.bn是空(None)
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)    # 普通卷积，且不使用激活函数
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False) # 1*1卷积(逐点卷积)
        # k=3且p=1，因此p=1-3//2=0，即p为0
        # 1*1卷积不会改变特征图的H值和W值，填充值可以直接给0，但是为了防范一些边界效应，使用p=(p-k//2)作为1*1卷积的填充
        # 边界效应即由于卷积操作会在输入特征图的边界处引入一些额外的值，从而导致输出特征图的边缘像素与内部像素存在差异

    def forward_fuse(self, x):  # (无BN且但分支)，推理时使用，已经融合后的参数
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):   # (有BN且三分支)，训练时使用，未融合参数
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        # 当self.bn(x)不为空时，id_out为self.bn(x)，相当于经过了BN层的残差连接
        # 如果self.bn为空，id_out为0
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases.
            将不同卷积分支的参数融合，把三个分支变成一个简单的卷积分支，减少参数，节省计算资源
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)   # 3*3卷积分支内部的卷积层参数和BN层参数融合
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)   # 1*1卷积分支内部的卷积层参数和BN层参数融合
        kernelid, biasid = self._fuse_bn_tensor(self.bn)    # BN层内部的卷积层参数和BN层融合
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
        # 返回一个新权重gamma_new和新偏差beta_new，二者都是将三个分支的权重/偏差相加得到

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor.
            将卷积核从1*1扩张成3*3
            为了方便融合操作，将不同大小的卷积核统一为相同的大小
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])
            # 在kernel1x1的上下左右四个方向向上填充1个像素，成了3*3

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network.
            将BN层参数融合进Conv层参数
            _fuse_bn_tensor函数接收一个参数branch，表示需要融合的网络分支，该branch可以是self.conv1这样的Conv类实例
                该函数的功能是根据输入的branch信息，生成合适的卷积核和偏置
        """
        if branch is None:  # 没有分支需要融合，直接返回0和0
            return 0, 0
        if isinstance(branch, Conv):    # 如果branch层是Conv类(包括卷积层、BN层、激活层)
        # 从branch中取出Conv层和BatchNorm层对应的属性值，并保存在本函数(_fuse_bn_tensor)的变量中
            kernel = branch.conv.weight # 卷积层的权重
            # kernel是一个大小为(out_channels, in_channels//groups, kernel_size[0], kernel_size[1])的张量
            # 将所有组的滤波器权重在out_channels方向堆叠在一起，最后的权重即为如上所示
            running_mean = branch.bn.running_mean   # BatchNorm层的运行平均值
            running_var = branch.bn.running_var # BatchNorm层的运行方差
            gamma = branch.bn.weight    # BatchNorm层的权重
            beta = branch.bn.bias   # BatchNorm层的偏置
            eps = branch.bn.eps # BatchNorm层的微小值，防止方差分母为0
        elif isinstance(branch, nn.BatchNorm2d):    # 如果branch是nn.BatchNorm2d类(只有BN层)
            if not hasattr(self, "id_tensor"):  # 创建单位矩阵
                # 单位矩阵的作用是表示一个恒等映射(identity mapping)，即输入数据直接传递到输出，相当于实现了残差连接
                # 检查类本身是否拥有属性id_tensor(单位矩阵)，如果没有的话执行下面的代码块(继续创建一个单位矩阵id_tensor)
                input_dim = self.c1 // self.g   # 根据输入通道与分组数计算单位矩阵大小
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                # 创建与Conv类的权重相同大小的权重
                for i in range(self.c1):    # 通过循环遍历将指定位置的元素设置为1
                    kernel_value[i, i % input_dim, 1, 1] = 1
                    # i%input_dim指i对input_dim取余，在遍历input_dim的同时，让该维度不超过input_dim
                    # 即将(out_channels, in_channels//groups, 1, 1)每一个滤波器的中心位置设置为1，其他值设置为0
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
                # 将numpy数组转换为了pytorch张量，并把pytorch张量移动到与分支权重张量相同的设备上
                # 确保张量位于同一设备上，以便高效计算
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        # 校正权重与偏差，将BN层的参数融合进Conv层
        # BN层的参数在训练过程中可以计算出来，在此处将计算出的BN参数与卷积参数融合，得到新的卷积参数
        # 而推理时不需要重新计算一遍BN的参数，直接使用融合后的参数就可以实现BN的功能，新权重和新偏差将用于推理时的卷积操作。
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)  # 将t变量重塑为了(-1,1,1,1)的大小，即广播张量
        # 广播张量允许形状不同的张量进行元素级乘法，重塑后的t可以任意与kernel相乘。如果不重塑，二者的惩罚将导致维度不匹配和错误
        return kernel * t, beta - running_mean * gamma / std    # 函数返回权重校正值gamma'和偏差校正值beta'
        # 在训练过程中计算上述的内容，这是融合前的；融合后gamma'和beta'成了已知的，在推理时直接拿来算，省去很多计算步骤

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class.
            合并两个卷积层(self.conv1和self.conv2)成单个卷积层(self.conv)，并移除合并过程中不再使用的属性
            删除未使用属性，可释放额外的内存空间，进一步降低网络的内存占用，这对于大型或复杂网络尤为重要；
            还可以简化代码结构，使其更容易理解和维护。
        """
        if hasattr(self, "conv"):   # 如果有conv属性，合并过程可能已经完成，函数直接返回
            return
        kernel, bias = self.get_equivalent_kernel_bias()    # 融合分支，得到等价的权重和偏差
        self.conv = nn.Conv2d(  # 创建新的卷积层并传参，使用的都是conv1的参数，conv1中conv取出的属性都是用户在创建RepConv中初始化的
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False) # requires_grad_属性设置为False，表示其权重在训练过程中不会更新
        self.conv.weight.data = kernel  # 将融合分支的权重和偏差分别赋值给新创建的卷积层的权重和偏差
        self.conv.bias.data = bias
        for para in self.parameters():  # 分离计算图，并删除无用的属性
            para.detach_()  # detach_方法分离属性与计算图的连接，参数的内存分配不再由图管理，可以安全释放
            # 如果参数在网络中不再使用，就会成为不必要的负担，可能导致内存泄漏，即网络保留了未使用的内存资源，可能会导致性能问题和不稳定
        self.__delattr__("conv1")   # __delattr__方法删除不再需要的属性，包括conv1、conv2、nm、bn、id_tensor
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:  # channels输入张量的通道数
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1) # 自适应平均池化层，将输入特征图缩小到空间尺寸为1*1
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True) # 用卷积层实现全连接层的功能
        # 用于将池化后的特征图中的信息转换为通道注意力权重。(输入输出通道数相同，内核大小1*1，无填充，带有偏置项)
        self.act = nn.Sigmoid() # 将注意力权重压缩至0-1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))
        # self.act(self.fc(self.pool(x)))得到注意力权重。
        # 注意力权重以逐元素乘法的方式与原始特征图x相乘，相当于放大模型关注的重要的通道信息，同时削弱不重要的通道信息


class SpatialAttention(nn.Module):
    """Spatial-attention module.
        空间注意力机制，帮助模型在处理图像时更加关注重要的空间信息
    """

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"  # 断言，规定卷积核大小一定为3或7
        padding = 3 if kernel_size == 7 else 1  # k=3,p=1或者k=7,p=3保证卷积操作后特征图的H和W不变
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)    # 输入通道2输出通道1的普通卷积
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))
        # 输入张量大小为(B,C,H,W)
        # torch.mean(x,1,keepdim=True)返回x在通道维度的平均值，输出为(B,1,H,W)
        # torch.max(x,1,keepdim=True)[0]返回x的通道最大值，输出为(B,1,H,W).(抛弃了[1]的通道最大值索引)
        # torch.cat操作将平均值和最大值连接，得到(B,2,H,W)
        # 再将(B,2,H,W)送入卷积和激活，得到(B,1,H,W)的空间注意力权重值。卷积是为了浓缩空间信息，激活是为了映射
        # 最后将权重与x进行逐元素相乘，放大模型关注的重要的空间信息，削弱不重要的空间信息


class CBAM(nn.Module):
    """Convolutional Block Attention Module.
        先通过通道注意力，再通过空间注意力
    """

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension.
        通道连接，沿指定的维度
    """

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension  # 指定了连接张量的维度
        # self.d=1表示沿通道维度进行连接
        # self.d=2表示沿高度维度进行连接

    def forward(self, x):   # x是一个包含要连接的张量列表的输入
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)
        # 返回x沿self.d指定的方向进行连接
