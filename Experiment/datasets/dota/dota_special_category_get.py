"""
检测同时包含某些特定类别目标的图片，提取出来
如container和large-vehicle
将他们的图片名存入到一个txt文件之中
txt中的每一行是一个图片名
"""
import os
import shutil

def find_images_with_targets_and_copy(annotation_dir, image_dir, target_classes, output_dir, result_file):
    images_with_targets = []

    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历注释文件夹
    for root, _, files in os.walk(annotation_dir):
        for file in files:
            if file.endswith('.txt'):
                txt_path = os.path.join(root, file)

                # 读取 txt 文件
                with open(txt_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # 检查是否包含目标类别
                categories = {line.split()[8] for line in lines if len(line.split()) > 8}
                if all(target in categories for target in target_classes):
                    # 获取对应的图片名称
                    image_name = os.path.splitext(file)[0] + '.jpg'  # 假设切割后的图片格式为 PNG
                    images_with_targets.append(image_name)

                    # 复制图片到输出文件夹
                    src_image_path = os.path.join(image_dir, image_name)
                    dst_image_path = os.path.join(output_dir, image_name)
                    if os.path.exists(src_image_path):
                        shutil.copy(src_image_path, dst_image_path)

    # 写入结果文件
    with open(result_file, 'w', encoding='utf-8') as f:
        for image in images_with_targets:
            f.write(image + '\n')

    return images_with_targets

# 数据集路径
annotation_dir = '/mnt/d/exp/sodaa_sob/datasets/test/labels/'  # DOTA标签文件夹
image_dir = '/mnt/d/exp/sodaa_sob/datasets/test/images/'  # 切割后的图片文件夹
output_dir = '/mnt/d/exp/sodaa_sob/datasets-similar-all/'  # 输出文件夹
result_file = '/mnt/d/exp/sodaa_sob/datasets-similar-all.txt'  # 结果文件

# 目标类别
target_classes = ['large-vehicle', 'container']

# 查找图片并复制
images = find_images_with_targets_and_copy(annotation_dir, image_dir, target_classes, output_dir, result_file)

# 输出结果
if images:
    print(f"包含目标类别 {target_classes} 的图片已复制到 {output_dir}, 结果已写入 {result_file}:")
    for img in images:
        print(img)
else:
    print(f"没有找到同时包含 {target_classes} 的图片。")

