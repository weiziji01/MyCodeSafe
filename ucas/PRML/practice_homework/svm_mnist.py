"""
Pattern Recognition And Machine Learning
SVM -- MNIST
Created by Weiziji on 2024/01/08
"""

from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score
from struct import unpack

# 读取图像
def readimage(path):
    with open(path,'rb') as f:
        magic,num,rows,cols=unpack('>4I',f.read(16))
        img=np.fromfile(f,dtype=np.uint8).reshape(num,784)
    return img

# 读取标签
def readlabel(path):
    with open(path,'rb') as f:
        magic,num=unpack('>2I',f.read(8))
        lab=np.fromfile(f,dtype=np.uint8)
    return lab

if __name__ == "__main__":
    # 数据加载
    train_data=readimage('/mnt/d/data/mnist/MNIST/raw/train-images-idx3-ubyte')
    train_label=readlabel('/mnt/d/data/mnist/MNIST/raw/train-labels-idx1-ubyte')
    test_data=readimage('/mnt/d/data/mnist/MNIST/raw/t10k-images-idx3-ubyte')
    test_label=readlabel('/mnt/d/data/mnist/MNIST/raw/t10k-labels-idx1-ubyte')

    # SVM模型训练
    svc=svm.SVC(kernel='linear',C=1)
    print("Training....")
    svc.fit(train_data,train_label)

    # 模型测试
    pred=svc.predict(test_data)
    print("svm_accruacy: ", accuracy_score(pred, test_label))
