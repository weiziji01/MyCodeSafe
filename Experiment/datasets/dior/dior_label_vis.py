"""
DIOR标注格式的标签可视化
"""
from xml.etree import ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def visualize_annotations(xml_path, image_folder):
    # 解析XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取基本信息
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    # 加载图像
    img_path = f"{image_folder}/{filename}"
    image = Image.open(img_path)

    # 创建绘图对象
    draw = ImageDraw.Draw(image)
    
    # 使用默认字体（如需中文支持需指定中文字体路径）
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    # 绘制每个对象的标注框
    for obj in root.findall('object'):
        name = obj.find('name').text
        robndbox = obj.find('robndbox')
        if robndbox is None:    # 水平框
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            # 绘制矩形框
            draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=2)
            draw.text((xmin, ymin-25), name, fill='red', font=font)
        else:   # 旋转框
            points = [
                (int(robndbox.find('x_left_top').text), int(robndbox.find('y_left_top').text)),
                (int(robndbox.find('x_right_top').text), int(robndbox.find('y_right_top').text)),
                (int(robndbox.find('x_right_bottom').text), int(robndbox.find('y_right_bottom').text)),
                (int(robndbox.find('x_left_bottom').text), int(robndbox.find('y_left_bottom').text)),
            ]
            # 绘制旋转框
            draw.polygon(points, outline='red', width=2)
            # 添加标签文本
            draw.text((points[0][0]+5, points[0][1]+5), name, fill='red', font=font)

    # 显示图像
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # 使用示例
    xml_path = "/mnt/d/data/dior-r/DIOR-R/Annotations/Oriented Bounding Boxes/00166.xml"  # XML文件路径
    image_folder = "/mnt/d/data/dior-r/DIOR-R/JPEGImages-trainval/"           # 图片所在目录
    visualize_annotations(xml_path, image_folder)
