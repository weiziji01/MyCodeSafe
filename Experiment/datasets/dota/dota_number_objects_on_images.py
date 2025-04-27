"""
读取DOTA标注的数据集中，包含某些特定类别的图像，将这些图片名（不带后缀）和带有多少目标输出
输出结果是在txt中，每一行的示例：00006__800__2600___1300,3
读取测试集中含有large-vehicle和container的图像，每个图像分别有多少对应的目标
"""
import os

def count_targets(label_dir, output_file):
    result_lines = []
    
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):  # 确保是 txt 文件
            file_path = os.path.join(label_dir, filename)
            image_name = os.path.splitext(filename)[0]  # 去掉扩展名
            large_vehicle_count = 0
            container_count = 0
            
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 9:
                        continue  # 确保数据格式正确
                    category = parts[8]  # 类别字段
                    if category == "large-vehicle":
                        large_vehicle_count += 1
                    elif category == "container":
                        container_count += 1
            
            total_count = large_vehicle_count + container_count
            
            if total_count >= 230:
                result_lines.append(f"{image_name},{total_count}")
    
    print(len(result_lines))
    
    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.write("\n".join(result_lines))

if __name__ == '__main__':
    # 示例使用
    label_folder = "/mnt/d/exp/sodaa_sob/datasets-similar/labels-all/"  # 修改为你的 DOTA 标签文件夹路径
    output_txt = "/mnt/d/exp/sodaa_sob/datasets-similar/similar_object_statistics_large_230.txt"
    count_targets(label_folder, output_txt)
    print(f"统计完成，结果保存在 {output_txt}")