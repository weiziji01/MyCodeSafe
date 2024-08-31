"""
将YOLO格式标签的数据集划分为训练集，验证集，测试集
"""
import os
import random
import shutil


# 创建保存数据的文件夹
def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def splitData(img_dir, label_dir, split_dir, train_pct, valid_pct, test_pct,TEST=True):
    '''
    args:
        img_dir:原图片数据集路径
        label_dir:原yolog格式txt文件数据集路径
        split_dir:划分后数据集保存路径
        TEST:是否划分测试集
        用于将数据集划分为YOLO数据集格式的训练集,验证集,测试集
    '''
    random.seed(42)  # 随机种子
    # 1.确定原图片数据集路径
    datasetimg_dir = img_dir
    # 确定原label数据集路径
    datasetlabel_dir = label_dir

    images_dir = os.path.join(split_dir, 'images')
    labels_dir = os.path.join(split_dir, 'labels')
    dir_list = [images_dir, labels_dir]
    if TEST:
        type_label = ['train', 'valid', 'test']
    else:
        type_label = ['train', 'valid']

    for i in range(len(dir_list)):
        for j in range(len(type_label)):
            makedir(os.path.join(dir_list[i], type_label[j]))

    # 3.确定将数据集划分为训练集，验证集，测试集的比例
    # train_pct = 0.8
    # valid_pct = 0.1
    # test_pct = 0.1
    # 4.划分
    labels = os.listdir(datasetlabel_dir)  # 展示目标文件夹下所有的文件名
    labels = list(filter(lambda x: x.endswith('.txt'), labels))  # 取到所有以.txt结尾的yolo格式文件
    random.shuffle(labels)  # 乱序路径
    label_count = len(labels)  # 计算图片数量
    train_point = int(label_count * train_pct)  # 0:train_pct
    valid_point = int(label_count * (train_pct + valid_pct))  # train_pct:valid_pct
    for i in range(label_count):
        if i < train_point:  # 保存0-train_point的图片和标签到训练集
            out_dir = os.path.join(images_dir, 'train')
            label_out_dir = os.path.join(labels_dir, 'train')

        elif train_point <= i < valid_point:  # 保存train_point-valid_point的图片和标签到验证集
            out_dir = os.path.join(images_dir, 'valid')
            label_out_dir = os.path.join(labels_dir, 'valid')
        else:  # 保存test_point-结束的图片和标签到测试集
            out_dir = os.path.join(images_dir, 'test')
            label_out_dir = os.path.join(labels_dir, 'test')

        label_target_path = os.path.join(label_out_dir, labels[i])  # 指定目标保存路径
        label_src_path = os.path.join(datasetlabel_dir, labels[i])  # 指定目标原文件路径
        img_target_path = os.path.join(out_dir, labels[i].split('.')[0] + '.JPG')  # 指定目标保存路径
        img_src_path = os.path.join(datasetimg_dir, labels[i].split('.')[0] + '.JPG')  # 指定目标原图像路径
        shutil.copy(label_src_path, label_target_path)  # 复制txt
        shutil.copy(img_src_path, img_target_path)  # 复制图片
        

    print('train:{}, valid:{}, test:{}'.format(train_point, valid_point - train_point,
                                               label_count - valid_point))


if __name__ == "__main__":
    img_dir = r'/mnt/d/exp/dji_car/datasets_label_txt/images'
    label_dir = r'/mnt/d/exp/dji_car/datasets_label_txt/labels'
    split_dir = r'/mnt/d/exp/dji_car/datasets'
    train_pct = 0.7
    val_pct = 0.2
    test_pct = 0.1

    splitData(img_dir, label_dir, split_dir, train_pct, val_pct, test_pct, TEST=True)
