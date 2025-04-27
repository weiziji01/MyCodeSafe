"""
返回文件夹内所有图片的尺寸大小统计
"""
import os
from PIL import Image
from collections import defaultdict

def get_image_size_distribution(folder_path):
    # 初始化尺寸统计字典
    size_counts = defaultdict(int)
    
    # 支持的图片扩展名
    valid_exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
    
    # 遍历文件夹及其子目录
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in valid_exts:
                file_path = os.path.join(root, filename)
                try:
                    # 打开图片并获取尺寸
                    with Image.open(file_path) as img:
                        size = img.size  # (width, height)
                        size_counts[size] += 1
                except Exception as e:
                    print(f"无法处理文件 {file_path}: {str(e)}")
    
    # 按出现次数排序
    sorted_sizes = sorted(size_counts.items(), key=lambda x: x[1], reverse=True)
    
    # 打印结果
    print("\n图片尺寸分布统计：")
    print("====================")
    for size, count in sorted_sizes:
        print(f"尺寸 {size[0]}x{size[1]}：共 {count} 张")
    print("====================")
    print(f"总计 {len(sorted_sizes)} 种不同尺寸")
    print(f"总图片数量 {sum(size_counts.values())} 张")

if __name__ == "__main__":
    # 替换为你的目标文件夹路径
    target_folder = "/mnt/d/data/visdrone2019/visdrone2019/val/images"
    
    # 检查路径有效性
    if not os.path.isdir(target_folder):
        print(f"错误：路径 {target_folder} 不存在或不是目录")
    else:
        get_image_size_distribution(target_folder)