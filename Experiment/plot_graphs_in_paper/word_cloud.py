"""
词云图片生成
"""
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. 数据准备
# Applied Projects 关键词词频
applied_data = {
    "Deep Learning": 470,
    "Fusion": 264,
    "Signal Processing": 237,
    "Object Detection": 208,
    "Radar": 202,
    "Perception": 175,
    "Intelligence": 175,
    "Information Processing": 154,
    "Image Processing": 139,
    "Feature Extraction": 133
}

# Funded Projects 关键词词频
funded_data = {
    "Deep Learning": 62,
    "Signal Processing": 38,
    "Object Detection": 34,
    "Synthetic Aperture Radar (SAR)": 26,
    "Image Processing": 25,
    "Fusion": 25,
    "Hyperspectral": 21,
    "Multimodal": 20,
    "Feature Extraction": 18,
    "Localization": 17
}

# 2. 生成词云函数
def generate_wordcloud(data, title, output_filename):
    # 创建词云对象
    wordcloud = WordCloud(
        width=800,  # 图像宽度
        height=400,  # 图像高度
        background_color="white",  # 背景颜色
        colormap="viridis",  # 蓝绿色系配色
        # font_path="arial.ttf",  # 字体文件路径（Arial字体）
        max_words=100,  # 最大词数
        max_font_size=150,  # 最大字体大小
        random_state=42  # 随机种子
    )

    # 生成词云
    wordcloud.generate_from_frequencies(data)

    # 显示词云
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")  # 关闭坐标轴
    plt.title(title, fontsize=16, pad=20)  # 添加标题
    plt.show()

    # 保存词云图
    wordcloud.to_file(output_filename)

# 3. 生成 Applied Projects 词云
generate_wordcloud(applied_data, 
                   "Applied Projects Word Cloud (English)", 
                   "applied_projects_wordcloud.png")

# 4. 生成 Funded Projects 词云
generate_wordcloud(funded_data, 
                   "Funded Projects Word Cloud (English)", 
                   "funded_projects_wordcloud.png")