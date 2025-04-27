"""
打印文件夹内所有文件的名称
"""
import os

def print_filenames_in_directory(directory_path):
    # 检查路径是否存在
    if not os.path.exists(directory_path):
        print(f"路径 '{directory_path}' 不存在")
        return
    
    # 检查是否是文件夹
    if not os.path.isdir(directory_path):
        print(f"'{directory_path}' 不是文件夹")
        return
    
    print(f"文件夹 '{directory_path}' 中的文件:")
    
    # 遍历文件夹中的所有条目
    for entry in os.listdir(directory_path):
        # 获取完整路径
        full_path = os.path.join(directory_path, entry)
        
        # 检查是否是文件（而不是子文件夹）
        if os.path.isfile(full_path):
            print(entry)

# 使用示例 - 替换为你想查看的文件夹路径
folder_path = "/mnt/d/learning/sca_temp_calc"  # 当前目录
print_filenames_in_directory(folder_path)