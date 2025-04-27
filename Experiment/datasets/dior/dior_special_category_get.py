"""
检测同时包含某些特定类别目标的图片，提取出来
如groundtrackfield和stadium
将他们的图片名存入到一个txt文件之中
txt中的每一行是一个图片名
同时复制那些图片至一个新的文件夹中
"""
import os
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm

def process_dior_dataset(annotations_dir, images_dir, output_root_dir, output_txt):
    """
    处理DIOR-R数据集，提取包含特定类别的图片和标注
    
    Args:
        annotations_dir: 原始标注文件(XML)所在文件夹路径
        images_dir: 原始图片所在文件夹路径
        output_root_dir: 输出根目录（将在此目录下创建images和annotations子文件夹）
        output_txt: 结果记录文本文件路径
    """
    # 创建输出子目录
    output_img_dir = os.path.join(output_root_dir, "images")
    output_anno_dir = os.path.join(output_root_dir, "annotations")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_anno_dir, exist_ok=True)
    
    # 目标类别
    target_classes = {'groundtrackfield', 'stadium'}
    
    # 收集符合条件的文件
    matched_files = set()  # 存储不带扩展名的文件名
    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
    
    print("正在扫描标注文件...")
    for xml_file in tqdm(xml_files):
        xml_path = os.path.join(annotations_dir, xml_file)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 检查所有object
            for obj in root.findall('object'):
                obj_class = obj.find('name').text
                if obj_class in target_classes:
                    # 获取不带扩展名的文件名
                    base_name = os.path.splitext(xml_file)[0]
                    matched_files.add(base_name)
                    break  # 找到一个目标类别即可
                    
        except Exception as e:
            print(f"\n处理文件 {xml_file} 时出错: {str(e)}")
            continue
    
    # 写入结果文件
    print("\n正在写入结果...")
    with open(output_txt, 'w') as f:
        for base_name in sorted(matched_files):
            f.write(f"{base_name}\n")
    
    # 复制文件
    print("正在复制文件...")
    copied_pairs = 0
    
    for base_name in tqdm(matched_files):
        # 复制图片文件（尝试不同扩展名）
        img_copied = False
        for img_ext in ['.jpg', '.jpeg', '.png']:
            img_src = os.path.join(images_dir, base_name + img_ext)
            if os.path.exists(img_src):
                img_dst = os.path.join(output_img_dir, base_name + img_ext)
                try:
                    shutil.copy2(img_src, img_dst)
                    img_copied = True
                    break
                except Exception as e:
                    print(f"\n无法复制图片 {base_name}: {str(e)}")
        
        # 复制标注文件
        xml_src = os.path.join(annotations_dir, base_name + '.xml')
        if os.path.exists(xml_src):
            xml_dst = os.path.join(output_anno_dir, base_name + '.xml')
            try:
                shutil.copy2(xml_src, xml_dst)
                if img_copied:
                    copied_pairs += 1
            except Exception as e:
                print(f"\n无法复制标注 {base_name}.xml: {str(e)}")
        else:
            print(f"\n未找到标注文件 {base_name}.xml")
    
    # 输出统计信息
    print("\n处理完成!")
    print(f"找到包含目标类别的文件数量: {len(matched_files)}")
    print(f"成功复制的图片-标注对数量: {copied_pairs}")
    print(f"图片已复制到: {output_img_dir}")
    print(f"标注已复制到: {output_anno_dir}")
    print(f"文件列表已保存到: {output_txt}")

if __name__ == "__main__":
    # 配置路径 (请根据实际情况修改)
    annotations_dir = "/mnt/d/data/dior-r/DIOR-R/Annotations/filtered_annotations"  # XML标注文件夹
    images_dir = "/mnt/d/data/dior-r/DIOR-R/JPEGImages-trainval/"        # 原始图片文件夹
    output_root_dir = "/mnt/d/exp/dior-exp/datasets-small/"  # 输出根目录
    output_txt = "/mnt/d/exp/dior-exp/datasets-small/datasets-similar_list.txt"           # 结果记录文件
    
    # 执行处理
    process_dior_dataset(annotations_dir, images_dir, output_root_dir, output_txt)
