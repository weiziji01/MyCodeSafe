#
# ! 将文件夹内的rgb图像转为灰度图像

from PIL import Image
import os

def convert_images_to_grayscale(input_folder, output_folder):
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(input_folder):
        # 计算对应的输出文件夹路径
        relative_path = os.path.relpath(root, input_folder)
        output_dir = os.path.join(output_folder, relative_path)

        # 如果输出文件夹不存在，则创建
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 处理当前文件夹中的所有文件
        for filename in files:
            # 检查文件是否为图像文件（以常见扩展名过滤）
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.JPG')):
                # 打开图像
                img_path = os.path.join(root, filename)
                img = Image.open(img_path)

                # 转换为灰度图像
                grayscale_img = img.convert('L')

                # 保存到对应的输出文件夹
                output_path = os.path.join(output_dir, filename)
                grayscale_img.save(output_path)
                print(f"Converted {filename} to grayscale and saved to {output_path}")

if __name__ == "__main__":
    input_folder = '/mnt/d/exp/dji_car/datasets/'   # 替换为你的输入文件夹路径
    output_folder = '/mnt/d/exp/dji_car/dji_car_allcolor/' # 替换为你的输出文件夹路径
    convert_images_to_grayscale(input_folder, output_folder)
