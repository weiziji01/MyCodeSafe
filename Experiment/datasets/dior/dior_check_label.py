"""
自动检测DIOR-R数据集中可能存在的问题标注：
    1. 宽或高接近0的无效框（面积接近为0）
    2. 宽或高小于1的尺寸框
并支持自动过滤问题标注并保存新文件
提供清晰的数据增强建议
使用几何变换精确计算旋转后的实际尺寸
保留原始XML结构，仅移除问题标注
"""
import xml.etree.ElementTree as ET
import math
import os
from tqdm import tqdm


def check_and_filter_annotations(xml_path, output_dir=None):
    """检查并过滤无效标注框
    Args:
        xml_path (str): 原始标注文件路径
        output_dir (str): 过滤后的输出目录（None表示仅检查不保存）
    Returns:
        tuple: (无效框数量, 小尺寸框数量, 总框数量)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    invalid_count = 0
    small_count = 0
    total_count = 0
    problematic_objects = []

    for obj in root.findall('object'):
        # 只检查robndbox类型的标注
        if obj.find('type').text != 'robndbox':
            continue

        total_count += 1
        robndbox = obj.find('robndbox')
        angle = float(obj.find('angle').text)

        # 提取四个角点坐标
        try:
            coords = [
                float(robndbox.find(tag).text)
                for tag in [
                    'x_left_top', 'y_left_top',
                    'x_right_top', 'y_right_top',
                    'x_right_bottom', 'y_right_bottom',
                    'x_left_bottom', 'y_left_bottom'
                ]
            ]
        except AttributeError as e:
            print(f"坐标缺失 @ {xml_path}: {e}")
            continue

        # 将坐标转换为四个点
        points = [
            (coords[0], coords[1]),
            (coords[2], coords[3]),
            (coords[4], coords[5]),
            (coords[6], coords[7])
        ]

        # 计算旋转后的实际宽高
        width, height = calculate_rotated_size(points, angle)

        # 检查标注问题
        if width <= 1e-5 or height <= 1e-5:
            invalid_count += 1
            problematic_objects.append(obj)
        elif width < 1.0 or height < 1.0:
            small_count += 1
            problematic_objects.append(obj)

    # 移除问题标注
    for obj in problematic_objects:
        root.remove(obj)

    # 保存过滤后的文件
    if output_dir:
        filename = os.path.basename(xml_path)
        output_path = os.path.join(output_dir, filename)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)

    return invalid_count, small_count, total_count

def calculate_rotated_size(points, angle):
    """计算旋转框实际尺寸"""
    # 计算中心点
    cx = sum(x for x, y in points) / 4
    cy = sum(y for x, y in points) / 4
    
    # 反向旋转角度
    theta = math.radians(-angle)
    
    # 旋转所有点
    rotated_points = []
    for x, y in points:
        dx = x - cx
        dy = y - cy
        new_x = dx * math.cos(theta) - dy * math.sin(theta) + cx
        new_y = dx * math.sin(theta) + dy * math.cos(theta) + cy
        rotated_points.append((new_x, new_y))
    
    # 计算宽高
    xs = [p[0] for p in rotated_points]
    ys = [p[1] for p in rotated_points]
    return max(xs)-min(xs), max(ys)-min(ys)


def process_folder(input_dir, output_dir=None):
    """批量处理整个文件夹的标注文件
    Args:
        input_dir (str): 原始标注文件夹路径
        output_dir (str): 过滤后的输出目录（None表示仅检查不保存）
    Returns:
        dict: 包含统计信息的字典
    """
    # 创建输出目录
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 收集所有XML文件
    xml_files = [f for f in os.listdir(input_dir) if f.endswith('.xml')]
    
    total_stats = {
        'total_files': len(xml_files),
        'total_boxes': 0,
        'invalid_boxes': 0,
        'small_boxes': 0,
        'problematic_files': []
    }

    # 添加进度条
    for filename in tqdm(xml_files, desc="Processing XML files"):
        xml_path = os.path.join(input_dir, filename)
        try:
            # 处理单个文件
            invalid, small, total = check_and_filter_annotations(
                xml_path,
                output_dir=output_dir
            )
            
            # 更新统计信息
            total_stats['total_boxes'] += total
            total_stats['invalid_boxes'] += invalid
            total_stats['small_boxes'] += small
            
            # 记录有问题的文件
            if invalid > 0 or small > 0:
                total_stats['problematic_files'].append({
                    'filename': filename,
                    'invalid': invalid,
                    'small': small,
                    'total': total
                })
                
        except Exception as e:
            print(f"\n处理文件 {filename} 时发生错误：{str(e)}")
            continue
            
    return total_stats


if __name__ == "__main__":
    input_folder = "/mnt/d/data/dior-r/DIOR-R/Annotations/Oriented Bounding Boxes/"
    output_folder = "/mnt/d/data/dior-r/DIOR-R/Annotations/filtered_annotations"  # 设为None则不保存
    
    stats = process_folder(input_folder, output_folder)
    
    # 打印统计报告
    print("\n检测报告：")
    print(f"已处理文件总数：{stats['total_files']}")
    print(f"总标注框数量：{stats['total_boxes']}")
    print(f"发现无效框数量：{stats['invalid_boxes']}")
    print(f"发现小尺寸框数量：{stats['small_boxes']}")
    print(f"存在问题的文件数量：{len(stats['problematic_files'])}")
    
    # 打印有问题的文件详情（前10个）
    if len(stats['problematic_files']) > 0:
        print("\n问题文件示例（最多显示10个）：")
        for file_info in stats['problematic_files'][:10]:
            print(f"文件：{file_info['filename']}")
            print(f"  - 总标注框：{file_info['total']}")
            print(f"  - 无效框：{file_info['invalid']}")
            print(f"  - 小尺寸框：{file_info['small']}")
            
    # 保存完整问题文件列表
    if stats['problematic_files']:
        with open("problematic_files.txt", "w") as f:
            for file_info in stats['problematic_files']:
                f.write(f"{file_info['filename']}\n")
        print("\n完整问题文件列表已保存到：problematic_files.txt")
        