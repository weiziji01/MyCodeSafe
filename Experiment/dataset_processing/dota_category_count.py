#
# ! dota格式数据集的类别实例数目统计
import os
from collections import defaultdict

def count_dota_categories(annotation_dir):
    # 字典用于存储类别及其对应目标数量
    category_count = defaultdict(int)

    # 遍历标注目录下的所有标注文件
    for annotation_file in os.listdir(annotation_dir):
        if annotation_file.endswith('.txt'):
            file_path = os.path.join(annotation_dir, annotation_file)
            with open(file_path, 'r') as f:
                for line in f:
                    # DOTA 标注格式最后一列为类别
                    data = line.strip().split()
                    if len(data) >= 9:  # 检查是否有足够的字段
                        category = data[8]
                        category_count[category] += 1

    return category_count

def print_category_statistics(category_count):
    print("Category Statistics:")
    for category, count in category_count.items():
        print(f"Category: {category}, Count: {count}")

# 示例使用
annotation_dir = '/mnt/d/exp/sodaa_sob/datasets/train/labels/'  # 替换为你的标注文件夹路径
category_count = count_dota_categories(annotation_dir)
print_category_statistics(category_count)

# SODAA-Train-split:Category Statistics:
# Category: airplane, Count: 20542
# Category: small-vehicle, Count: 357269
# Category: ship, Count: 39569
# Category: container, Count: 108637
# Category: large-vehicle, Count: 14341
# Category: storage-tank, Count: 26919
# Category: windmill, Count: 16786
# Category: swimming-pool, Count: 22712
# Category: helicopter, Count: 1084
