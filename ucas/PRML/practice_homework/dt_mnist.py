"""
Pattern Recognition And Machine Learning
Decision Tree -- MNIST
Created by Weiziji on 2024/01/08
"""

from sklearn import tree
from sklearn.metrics import confusion_matrix
import numpy as np
from struct import unpack
import matplotlib.pyplot as plt
import time
# 内置struct处理存储在文件中的二进制数据，unpack实现二进制数据转换

# 读取图像
def readimage(path):
    with open(path,'rb') as f:  # 读取二进制文件
        magic,num,rows,cols=unpack('>4I',f.read(16))
        # magic用来标示文件格式，默认不变为2051
        # num输出图像数量
        # rows与cols分别代表每张图像的行数与列数
        # >4I:>代表存储方向，>4I是指需要前4个整数；16是指读取4个整数共16个字节
        img=np.fromfile(f,dtype=np.uint8).reshape(num,784)
        # dtype:返回数组的数据类型
        # img中每行代表一张图像，因为每行有28*28=784个元素
    return img

# 读取标签
def readlabel(path):
    with open(path,'rb') as f:
        magic,num=unpack('>2I',f.read(8))
        lab=np.fromfile(f,dtype=np.uint8)
    return lab

if __name__ == "__main__":
    start=time.perf_counter()
    # 数据加载
    train_data=readimage('/mnt/d/data/mnist/MNIST/raw/train-images-idx3-ubyte')
    train_label=readlabel('/mnt/d/data/mnist/MNIST/raw/train-labels-idx1-ubyte')
    test_data=readimage('/mnt/d/data/mnist/MNIST/raw/t10k-images-idx3-ubyte')
    test_label=readlabel('/mnt/d/data/mnist/MNIST/raw/t10k-labels-idx1-ubyte')

    # 构建决策树
    model=tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=10, 
                                           min_samples_split=3)
    # criterion特征选择标准设置为信息增益
    # splitter特征点划分标准为最好
    # 最大层深度不做设置
    # 内部节点再划分所需样本数设置为3
    print("Training...")
    model.fit(train_data,train_label)  # 训练
    test_pred=model.predict(test_data)
    score=model.score(test_data,test_label)    # 测试
    print(score*100,"%")

    # 计算混淆矩阵
    C=confusion_matrix(test_label,test_pred)
    plt.matshow(C,cmap=plt.cm.Reds)
    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j,i],xy=(i,j),
                         horizontalalignment='center', 
                         verticalalignment='center')
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(range(0,10))
    plt.yticks(range(0,10))
    plt.savefig("decision_tree_cm.png")

    # 可视化决策树
    text_pre=tree.export_text(model)
    with open("decision_tree.log","w") as f:
        f.write(text_pre)

    # 计算耗时
    end=time.perf_counter()
    print("Totally time: ", end-start)
