"""
绘制气泡图，或者叫做思维导图，以SODA-A论文中的小目标检测方法为例
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# 创建有向图
G = nx.DiGraph()

# 中心节点
central_node = "Small Object Detection"
G.add_node(central_node)

# 定义类别及其方法
categories = {
    "Attention-Based": ["KB-RANN", "SCRDet", "FBR-Net", "AF-SSD", "MSCCA", "CANet"],
    "Context-Modeling": ["RCNN for SOD", "PyramidBox", "SINet", "R2-CNN", "FS-SSD", "CAD-Net", "CAB Net"],
    "Feature-Imitation": ["Super-Resolution-based", "Similarity Learning"],
    "Sample-Oriented": ["Data Augmentation", "Label Assignment"],
    "Scale-Aware": ["Scale-Specific Detectors", "Feature Fusion"],
    "Focus-and-Detect": ["ClusDet", "CDMNet", "DMNet", "CRENet", "Adazoom", "PRDet", "F&D"]
}

# 颜色方案（淡雅配色）
category_colors = {
    "Attention-Based": "#6495ED",  # 淡蓝
    "Context-Modeling": "#FF6F61",  # 淡红
    "Feature-Imitation": "#9ACD32",  # 淡绿
    "Sample-Oriented": "#66CDAA",  # 淡青
    "Scale-Aware": "#20B2AA",  # 深青
    "Focus-and-Detect": "#4169E1",  # 深蓝
}

# 添加节点和边
for category, methods in categories.items():
    G.add_node(category)
    G.add_edge(central_node, category)  # 连接中心节点

    for method in methods:
        G.add_node(method)
        G.add_edge(category, method)  # 连接类别

# 生成径向布局
pos = nx.circular_layout(G)

# **手动调整中心节点**
pos[central_node] = np.array([0, 0])  # 中心点固定在原点

# **手动调整类别节点位置**
angle_step = 2 * np.pi / len(categories)  # 平均分布角度
radius = 0.6  # 类别节点的半径
i = 0
for category in categories.keys():
    angle = i * angle_step
    pos[category] = np.array([radius * np.cos(angle), radius * np.sin(angle)])
    i += 1

# **手动调整方法节点位置**
for category, methods in categories.items():
    base_pos = pos[category]  # 类别节点位置
    method_radius = 0.3  # 方法节点的半径
    angle_step = 2 * np.pi / len(methods) if methods else 0

    for j, method in enumerate(methods):
        angle = j * angle_step
        pos[method] = np.array([
            base_pos[0] + method_radius * np.cos(angle),
            base_pos[1] + method_radius * np.sin(angle)
        ])

# 绘制图形
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)

# 绘制连接线（使用大箭头）
for edge in G.edges:
    start, end = pos[edge[0]], pos[edge[1]]
    arrow = plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], 
                      head_width=0.03, head_length=0.03, fc="gray", ec="gray", alpha=0.7)
    
# 绘制节点（大气泡）
for category, color in category_colors.items():
    # 绘制类别气泡
    x, y = pos[category]
    ax.text(x, y, category, fontsize=12, fontweight='bold', fontname="Microsoft YaHei",
            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", facecolor=color, edgecolor=color))

    # 绘制方法气泡
    for method in categories[category]:
        x, y = pos[method]
        ax.text(x, y, method, fontsize=10, fontweight='bold', fontname="Microsoft YaHei",
                ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color))

# 绘制中心节点
x, y = pos[central_node]
ax.text(x, y, central_node, fontsize=14, fontweight='bold', fontname="Microsoft YaHei",
        ha='center', va='center', bbox=dict(boxstyle="round,pad=0.8", facecolor="gray", edgecolor="black", alpha=0.8))

# 移除坐标轴
plt.axis("off")

# 显示图形
plt.title("Small Object Detection Overview", fontsize=16, fontweight='bold', fontname="Microsoft YaHei")
plt.show()

