#
# ! 将SODAA数据集里的images文件夹下的图片按照Annotations的形式划分

import os
import shutil
import json

def organize_images_by_annotations(images_folder, annotations_folder, output_folder, copy_images=True):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历Annotations文件夹中的所有子文件夹
    for subfolder in os.listdir(annotations_folder):
        subfolder_path = os.path.join(annotations_folder, subfolder)

        # 确保当前路径是一个目录
        if os.path.isdir(subfolder_path):
            # 为每个子文件夹创建对应的输出文件夹
            output_subfolder = os.path.join(output_folder, subfolder)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)

            # 遍历子文件夹中的所有JSON文件
            for json_file in os.listdir(subfolder_path):
                if json_file.endswith('.json'):
                    json_path = os.path.join(subfolder_path, json_file)
                    image_filename = json_file.replace('.json', '.jpg')  # 假设图像格式为.jpg
                    image_src_path = os.path.join(images_folder, image_filename)

                    # 如果图像文件存在，则将其复制或移动到对应的文件夹
                    if os.path.exists(image_src_path):
                        image_dst_path = os.path.join(output_subfolder, image_filename)
                        if copy_images:
                            shutil.copy(image_src_path, image_dst_path)
                            print(f"Copied {image_filename} to {output_subfolder}")
                        else:
                            shutil.move(image_src_path, image_dst_path)
                            print(f"Moved {image_filename} to {output_subfolder}")
                    else:
                        print(f"Image file {image_filename} not found in {images_folder}")

if __name__ == "__main__":
    # 使用示例
    images_folder = '/mnt/d/exp/sodaa_sob/dataset_unsplit/Images/'  # 替换为Images文件夹的路径
    annotations_folder = '/mnt/d/exp/sodaa_sob/dataset_unsplit/Annotations/'  # 替换为Annotations文件夹的路径
    output_folder = '/mnt/d/exp/sodaa_sob/dataset_unsplit/Images_s/'  # 替换为输出文件夹的路径

    organize_images_by_annotations(images_folder, annotations_folder, output_folder, copy_images=True)
