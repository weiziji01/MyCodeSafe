#
# * 对很少的yolov8数据进行数据增强
# import albumentations as A
# import albumentations as A
import albumentations as A
import cv2
import os
from tqdm import tqdm

# 定义每个增强操作的变换和对应的名称后缀
augmentation_operations = [
    {
        'name': 'hflip',
        'transform': A.HorizontalFlip(p=1.0)
    },
    {
        'name': 'vflip',
        'transform': A.VerticalFlip(p=1.0)
    },
    {
        'name': 'rotate15',
        'transform': A.Rotate(limit=15, p=1.0)
    },
    {
        'name': 'rotate_neg15',
        'transform': A.Rotate(limit=-15, p=1.0)
    },
    {
        'name': 'brightness',
        'transform': A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)
    },
    {
        'name': 'contrast',
        'transform': A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.2, p=1.0)
    },
    # {
    #     'name': 'gauss_noise',
    #     'transform': A.GaussianNoise(var_limit=(10.0, 50.0), p=1.0)
    # },
    # 如果需要其他增强操作，可以在这里添加
]

def load_yolo_labels(label_path):
    """
    读取YOLO格式的标签文件
    """
    bboxes = []
    labels = []
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, width, height = parts
            bboxes.append([float(x_center), float(y_center), float(width), float(height)])
            labels.append(int(class_id))
    return bboxes, labels

def save_yolo_labels(label_path, bboxes, labels):
    """
    保存YOLO格式的标签文件
    """
    with open(label_path, 'w') as file:
        for cls, bbox in zip(labels, bboxes):
            file.write(f"{cls} {' '.join(map(str, bbox))}\n")

def augment_image(image, bboxes, labels, transform):
    """
    应用数据增强
    """
    augmented = transform(image=image, bboxes=bboxes, labels=labels)
    return augmented['image'], augmented['bboxes'], augmented['labels']

def main(images_dir, labels_dir, augmented_dir):
    """
    对整个数据集进行批量增强
    """
    augmented_images_dir = os.path.join(augmented_dir, 'images')
    augmented_labels_dir = os.path.join(augmented_dir, 'labels')
    os.makedirs(augmented_images_dir, exist_ok=True)
    os.makedirs(augmented_labels_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    for img_file in tqdm(image_files, desc="Augmenting images"):
        img_path = os.path.join(images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        # 读取图像（灰度图）
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"无法读取图像: {img_path}")
            continue
        # 将灰度图转换为三通道，因为某些增强操作可能需要多通道输入
        # 但如果确定所有增强操作都支持单通道，可以跳过这一步
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 读取标签
        if not os.path.exists(label_path):
            print(f"标签文件不存在: {label_path}")
            continue
        bboxes, labels = load_yolo_labels(label_path)
        
        for aug in augmentation_operations:
            transform = A.Compose([
                aug['transform']
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))
            
            try:
                augmented_image, augmented_bboxes, augmented_labels = augment_image(image, bboxes, labels, transform)
            except Exception as e:
                print(f"增强过程中出错: {e}，图像: {img_path}，增强操作: {aug['name']}")
                continue
            
            # 保存增强后的图像
            augmented_img_filename = f"{os.path.splitext(img_file)[0]}_{aug['name']}.jpg"
            augmented_img_path = os.path.join(augmented_images_dir, augmented_img_filename)
            cv2.imwrite(augmented_img_path, augmented_image)
            
            # 保存增强后的标签
            augmented_label_filename = f"{os.path.splitext(img_file)[0]}_{aug['name']}.txt"
            augmented_label_path = os.path.join(augmented_labels_dir, augmented_label_filename)
            save_yolo_labels(augmented_label_path, augmented_bboxes, augmented_labels)

if __name__ == "__main__":
    # 设置路径
    original_images_dir = '/mnt/d/exp/qiyuan/datasets_try/train/images/'       # 原始图像目录
    original_labels_dir = '/mnt/d/exp/qiyuan/datasets_try/train/labels/'       # 原始标签目录
    augmented_data_dir = '/mnt/d/exp/qiyuan/augmented_data'         # 增强后数据的保存目录
    
    # 执行增强
    main(original_images_dir, original_labels_dir, augmented_data_dir)
