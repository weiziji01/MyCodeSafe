"""
YOLO格式标注转为DOTA格式标注
"""

import argparse
import os
from PIL import Image


def convert_yolo_to_dota(label_dir, image_dir, classes_path, output_dir, image_ext):
    # 读取类别名称
    with open(classes_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    os.makedirs(output_dir, exist_ok=True)

    # 自动检测的图像扩展名列表
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    if image_ext:
        image_extensions = [image_ext.lower()]

    # 遍历所有YOLO标签文件
    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue

        label_path = os.path.join(label_dir, label_file)
        base_name = os.path.splitext(label_file)[0]

        # 查找对应的图像文件
        image_found = False
        for ext in image_extensions:
            image_path = os.path.join(image_dir, base_name + ext)
            if os.path.exists(image_path):
                image_found = True
                break

        if not image_found:
            print(f"⚠️ 未找到 {base_name} 的图片文件，跳过")
            continue

        try:
            # 获取图片尺寸
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"❌ 打开图片失败 {image_path}: {e}")
            continue

        # 读取YOLO标注内容
        with open(label_path, "r") as f:
            yolo_lines = f.readlines()

        dota_lines = []
        for line in yolo_lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                print(f"⚠️ 无效行格式 {label_file}: {line}")
                continue

            try:
                # 解析YOLO格式数据
                class_idx = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError:
                print(f"⚠️ 数值解析错误 {label_file}: {line}")
                continue

            # 验证类别索引有效性
            if class_idx < 0 or class_idx >= len(classes):
                print(f"⚠️ 无效类别索引 {class_idx} in {label_file}")
                continue

            # 转换为绝对坐标
            class_name = classes[class_idx]
            w = width * img_width
            h = height * img_height
            xc = x_center * img_width
            yc = y_center * img_height

            # 计算边界框坐标
            x0 = xc - w / 2
            y0 = yc - h / 2
            x1 = xc + w / 2
            y1 = yc + h / 2

            # 生成DOTA格式的四个顶点坐标（顺时针顺序）
            points = [
                (x0, y0),  # 左上
                (x1, y0),  # 右上
                (x1, y1),  # 右下
                (x0, y1),  # 左下
            ]

            # 四舍五入为整数并格式化坐标
            str_points = " ".join(
                [f"{int(round(x))} {int(round(y))}" for (x, y) in points]
            )
            dota_line = f"{str_points} {class_name} 0\n"
            dota_lines.append(dota_line)

        # 写入DOTA格式文件
        output_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(output_path, "w") as f:
            f.writelines(dota_lines)
        print(f"✅ 转换完成: {label_file} → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO格式转DOTA格式标注转换工具")
    parser.add_argument("--label_dir", required=True, help="YOLO标签文件目录")
    parser.add_argument("--image_dir", required=True, help="对应图片文件目录")
    parser.add_argument("--classes", required=True, help="类别定义文件路径")
    parser.add_argument("--output_dir", required=True, help="DOTA格式输出目录")
    parser.add_argument(
        "--image_ext", default="", help="图片文件扩展名（如未指定则自动检测）"
    )

    args = parser.parse_args()

    convert_yolo_to_dota(
        args.label_dir,
        args.image_dir,
        args.classes,
        args.output_dir,
        args.image_ext.strip("."),  # 自动去除可能包含的.
    )


