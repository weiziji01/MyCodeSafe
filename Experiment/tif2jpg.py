#
# ! 将tif文件转换为jpg格式的文件

from PIL import Image, ImageSequence
import os

# 增加最大像素限制
Image.MAX_IMAGE_PIXELS = None

def convert_tif_to_jpg(input_dir, output_dir, quality=85):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有 tif 文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            try:
                # 打开 tif 文件
                tif_image = Image.open(os.path.join(input_dir, filename))
                
                # 检查 tif 文件是否有多页
                for i, page in enumerate(ImageSequence.Iterator(tif_image)):
                    # 将文件名的后缀改为 jpg
                    jpg_filename = os.path.splitext(filename)[0] + (f'_page_{i}.jpg' if i > 0 else '.jpg')
                    
                    # 如果图像不是 RGB 模式，则转换为 RGB
                    if page.mode != 'RGB':
                        page = page.convert('RGB')
                    
                    # 保存为 jpg 格式，设置质量参数
                    page.save(os.path.join(output_dir, jpg_filename), 'JPEG', quality=quality)
                    print(f"Converted {filename} page {i} to {jpg_filename}")
                    
            except Exception as e:
                print(f"Error converting {filename}: {e}")

# 使用示例
input_directory = '/mnt/d/tif/'  # 替换为你的tif图片所在的文件夹路径
output_directory = '/mnt/d/jpg'  # 替换为你想存储jpg图片的文件夹路径
convert_tif_to_jpg(input_directory, output_directory)

