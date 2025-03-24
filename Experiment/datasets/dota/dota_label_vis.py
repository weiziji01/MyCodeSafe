"""
DOTA标签可视化在图像上
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_rotated_box(image, points, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制旋转矩形框
    :param image: 输入图像
    :param points: 四个顶点坐标，格式[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    :param color: 框颜色
    :param thickness: 线宽
    """
    points = np.array(points, dtype=np.int32)
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)
    return image

def draw_dota_labels(img_path, label_path, color_map=None, show=True):
    """
    可视化DOTA标注
    :param img_path: 图像路径
    :param label_path: 标签文件路径
    :param color_map: 自定义颜色映射字典
    :param show: 是否显示结果
    :return: 绘制后的图像
    """
    # 读取图像
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 默认颜色映射（可自定义修改）
    if color_map is None:
        color_map = {
            'pedestrian': (255, 0, 0),
            'people': (0, 255, 0),
            'bicycle': (0, 0, 255),
            'car': (255,255,0),
            'van': (255,0,255),
            'truck': (0,255,255),
            'tricycle': (128,0,128),
            'awning-tricycle': (128,128,0),
            'bus': (0,128,128),
            'motor': (128,0,128)
        }
    
    # 读取标签文件
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 解析每行标注
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        
        # 解析标签信息
        class_name = parts[8]
        points = list(map(float, parts[:8]))
        points = np.array(points, dtype=np.float32).reshape(4, 2)
        
        # 获取颜色
        color = color_map.get(class_name, (0, 255, 0))  # 默认绿色
        
        # 绘制旋转框
        image = draw_rotated_box(image, points, color=color)
        
        # 添加类别文本（在第一个点位置显示）
        text_pos = tuple(map(int, points[0]))
        cv2.putText(image, class_name, text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    if show:
        plt.figure(figsize=(12, 12))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    
    return image

# 使用示例 --------------------------------------------------
if __name__ == "__main__":
    # 修改为你的实际路径
    img_file = "/mnt/d/data/visdrone2019/visdrone2019/train/images/0000002_00005_d_0000014.jpg"
    label_file = "/mnt/d/data/visdrone2019/visdrone2019/train/labels/0000002_00005_d_0000014.txt"
    
    # 可视化标注
    result_image = draw_dota_labels(img_file, label_file)
    
    # 保存结果（可选）
    # result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("output.jpg", result_image)