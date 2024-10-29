#
# ! 将以YOLOv8格式标注的大图裁剪为小图

import cv2
import os

# 定义函数：将归一化坐标转换为像素坐标
def yolo_to_pixel(label, img_width, img_height):
    cls, x_center, y_center, width, height = label
    x_center_pixel = x_center * img_width
    y_center_pixel = y_center * img_height
    width_pixel = width * img_width
    height_pixel = height * img_height
    return cls, x_center_pixel, y_center_pixel, width_pixel, height_pixel

# 定义函数：将像素坐标转换为归一化YOLO格式
def pixel_to_yolo(cls, x_center, y_center, width, height, img_width, img_height):
    x_center_yolo = x_center / img_width
    y_center_yolo = y_center / img_height
    width_yolo = width / img_width
    height_yolo = height / img_height
    return cls, x_center_yolo, y_center_yolo, width_yolo, height_yolo

# 定义函数：检查目标是否在裁剪区域内
def is_in_crop(x_center, y_center, width, height, x_start, y_start, crop_size):
    x_min = x_center - width / 2
    x_max = x_center + width / 2
    y_min = y_center - height / 2
    y_max = y_center + height / 2
    return not (x_max < x_start or x_min > x_start + crop_size or
                y_max < y_start or y_min > y_start + crop_size)

# 加载大图像和标签文件
image_path = 'path_to_your_image.jpg'
label_path = 'path_to_your_label.txt'  # YOLOv8格式的标签文件路径
output_image_folder = 'cropped_images'  # 裁剪小图的输出文件夹
output_label_folder = 'cropped_labels'  # 裁剪小图对应标签的输出文件夹
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)

# 读取图像
image = cv2.imread(image_path)
image_height, image_width, _ = image.shape

# 读取YOLOv8标签
with open(label_path, 'r') as f:
    labels = [list(map(float, line.strip().split())) for line in f]

# 设置裁剪窗口的大小和步长
crop_size = 800
step_size = 250
crop_count = 0

# 开始裁剪
for y in range(0, image_height - crop_size + 1, step_size):
    for x in range(0, image_width - crop_size + 1, step_size):
        # 裁剪图像
        cropped_image = image[y:y+crop_size, x:x+crop_size]
        
        # 保存裁剪的图像
        crop_image_filename = f'crop_{crop_count}.jpg'
        crop_image_path = os.path.join(output_image_folder, crop_image_filename)
        cv2.imwrite(crop_image_path, cropped_image)
        
        # 处理标签
        crop_label_filename = f'crop_{crop_count}.txt'
        crop_label_path = os.path.join(output_label_folder, crop_label_filename)
        with open(crop_label_path, 'w') as label_file:
            for label in labels:
                # 将归一化的YOLO坐标转换为像素坐标
                cls, x_center_pixel, y_center_pixel, width_pixel, height_pixel = yolo_to_pixel(
                    label, image_width, image_height)
                
                # 检查目标是否在裁剪区域内
                if is_in_crop(x_center_pixel, y_center_pixel, width_pixel, height_pixel, x, y, crop_size):
                    # 更新裁剪后的小图中的坐标（相对于裁剪区域）
                    x_center_cropped = x_center_pixel - x
                    y_center_cropped = y_center_pixel - y
                    
                    # 重新归一化坐标
                    cls, x_center_yolo, y_center_yolo, width_yolo, height_yolo = pixel_to_yolo(
                        cls, x_center_cropped, y_center_cropped, width_pixel, height_pixel, crop_size, crop_size)
                    
                    # 写入标签文件
                    label_file.write(f'{cls} {x_center_yolo} {y_center_yolo} {width_yolo} {height_yolo}\n')
        
        crop_count += 1

print(f"裁剪和标签生成完成，总共处理了 {crop_count} 张小图。")
