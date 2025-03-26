"""
统计DIOR-R数据集中的角度分布，并绘制相关直方图
"""
import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def collect_angles_from_folder(xml_folder):
    """收集文件夹中所有XML文件的旋转角度"""
    angles = []
    xml_files = [f for f in os.listdir(xml_folder) if f.endswith('.xml')]
    
    for filename in tqdm(xml_files, desc="Processing XML files"):
        xml_path = os.path.join(xml_folder, filename)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                if obj.find('type').text == 'robndbox':
                    angle = float(obj.find('angle').text)
                    angles.append(angle)
                # 如果想要只统计某个类别的角度
                # if obj.find('name').text == 'airplane':  # 只统计飞机类别的角度
                #     angle = float(obj.find('angle').text)
                #     angles.append(angle)
                    
        except Exception as e:
            print(f"\nError processing {filename}: {str(e)}")
            continue
            
    return np.array(angles)

def plot_angle_distribution(angles, bin_size=5):
    """绘制角度分布直方图"""
    plt.figure(figsize=(12, 6))
    
    # 计算合适的bins
    min_angle, max_angle = np.min(angles), np.max(angles)
    bins = np.arange(min_angle, max_angle + bin_size, bin_size)
    
    # 绘制直方图
    n, bins, patches = plt.hist(angles, bins=bins, 
                               edgecolor='black', 
                               alpha=0.7,
                               color='#1f77b4')
    
    # 添加统计信息
    plt.title(f"DIOR-R Dataset Object Angle Distribution (n={len(angles)})", fontsize=14)
    plt.xlabel("Rotation Angle (degrees)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.grid(axis='y', alpha=0.5)
    
    # 添加数值标签
    for i in range(len(patches)):
        plt.text(patches[i].get_x() + patches[i].get_width()/2, 
                patches[i].get_height()+0.5, 
                str(int(n[i])), 
                ha='center', 
                va='bottom')
    
    # 保存图片
    plt.savefig('angle_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印基本统计信息
    print("\nAngle Statistics:")
    print(f"- Min angle: {min_angle:.2f}°")
    print(f"- Max angle: {max_angle:.2f}°")
    print(f"- Mean angle: {np.mean(angles):.2f}°")
    print(f"- Median angle: {np.median(angles):.2f}°")
    print(f"- Most common angle range: {bins[np.argmax(n)]:.1f}°-{bins[np.argmax(n)+1]:.1f}°")
    
def plot_polar_distribution(angles, bin_size=5):
    """极坐标角度分布图"""
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='polar')
    
    bins = np.arange(0, 360 + bin_size, bin_size)
    hist, _ = np.histogram(angles, bins=bins)
    theta = np.deg2rad(bins[:-1] + bin_size/2)
    
    bars = ax.bar(theta, hist, width=np.deg2rad(bin_size)*0.9,
                 bottom=0.0, color='#1f77b4', alpha=0.7)
    
    ax.set_title(f"Polar Angle Distribution (n={len(angles)})", pad=20)
    plt.savefig('polar_distribution.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # 配置路径
    xml_folder = "/mnt/d/data/dior-r/DIOR-R/Annotations/Oriented Bounding Boxes/"
    
    # 收集所有角度数据
    angles = collect_angles_from_folder(xml_folder)
    
    # 绘制直方图（可调整bin_size参数改变分组粗细）
    plot_angle_distribution(angles, bin_size=5)
    
    # 可选：保存原始角度数据
    np.save('angles_data.npy', angles)