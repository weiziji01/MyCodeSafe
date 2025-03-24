"""
查看DOTA格式文件夹内下的所有标注文件是否存在无效标注，如果有，则打印出来哪一个存在
无效标注包括以下情形：
1. 坐标数目不足或超出预期
2. 坐标值不是有效的数字
3. 坐标值出现负数或者超出合理范围
"""
import os

def is_valid_dota_annotation(line):
    parts = line.strip().split()
    if len(parts) < 9:
        return False  # 必须至少有 9 个元素
    
    try:
        coords = list(map(float, parts[:8]))  # 前 8 个应该是坐标
    except ValueError:
        return False  # 无法转换为浮点数
    
    if any(c < 0 for c in coords):
        return False  # 坐标不能是负数
    
    return True

def check_dota_annotations(folder_path):
    invalid_files = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):  # 只检查txt文件
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as f:
                for line_number, line in enumerate(f, start=1):
                    if not is_valid_dota_annotation(line):
                        print(f"无效标注: {file_name}, 行号: {line_number}")
                        invalid_files.append(file_name)
                        break  # 找到一个错误就跳过该文件
    
    if invalid_files:
        print("以下文件包含无效标注:")
        for file in invalid_files:
            print(file)
    else:
        print("所有标注文件均有效。")

# 使用示例
folder_path = "/mnt/d/data/visdrone2019/visdrone2019/test/labels"  # 替换为你的标注文件夹路径
check_dota_annotations(folder_path)