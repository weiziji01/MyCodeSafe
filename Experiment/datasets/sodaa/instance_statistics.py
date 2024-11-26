import os
import json
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from cityscapesscripts.helpers.csHelpers import colors
from matplotlib.colors import LinearSegmentedColormap


def process_annotation_file(file_path, category_count, area_stats, category_image_count):
    """
    处理单个标注文件，更新统计数据。
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    # 获取当前图像 ID
    image_id = data['images']['id']

    for annotation in data['annotations']:
        category_id = annotation['category_id']
        category_count[category_id] += 1

        # 更新该类别的图像集合
        category_image_count[category_id].add(image_id)

        # 统计面积数据
        area = annotation['area']
        if area is not None:  # 跳过无效面积
            area_stats[category_id].append(area)


def process_all_subfolders(root_folder):
    """
    遍历 rawAnnotations 下的 train、val 和 test 文件夹，统计所有标注文件中的信息。
    """
    category_count = defaultdict(int)
    area_stats = defaultdict(list)
    category_image_count = defaultdict(set)  # 使用集合来统计每个类别的图像 ID

    for subfolder in ['train', 'val', 'test']:
        folder_path = os.path.join(root_folder, subfolder)
        if not os.path.exists(folder_path):
            print(f"子文件夹 {subfolder} 不存在，跳过。")
            continue

        # 遍历子文件夹中的所有 JSON 文件
        for root, _, files in os.walk(folder_path):
            for file_name in files:
                if file_name.endswith('.json'):  # 只处理 JSON 文件
                    file_path = os.path.join(root, file_name)
                    print(f"正在处理文件: {file_path}")
                    process_annotation_file(file_path, category_count, area_stats, category_image_count)

    return category_count, area_stats, category_image_count


if __name__ == '__main__':
    # 主程序
    root_folder = "/mnt/d/exp/sodaa_sob/datasets/rawAnnotations/"  # 替换为 rawAnnotations 文件夹路径
    category_mapping = {
        0: "airplane", 1: "helicopter", 2: "small-vehicle", 3: "large-vehicle",
        4: "ship", 5: "container", 6: "storage-tank", 7: "swimming-pool",
        8: "windmill", 9: "ignore"
    }

    category_count, area_stats, category_image_count = process_all_subfolders(root_folder)

    # 打印统计结果
    print("\n统计结果:")
    for cat_id, count in category_count.items():
        print(f"{category_mapping.get(cat_id, '未知类别')} ({cat_id}): {count} 个标注")

    print("\n各类别面积统计:")
    for cat_id, areas in area_stats.items():
        print(f"{category_mapping.get(cat_id, '未知类别')} ({cat_id}):")
        print(f"  总面积: {sum(areas):.2f}")
        print(f"  平均面积: {sum(areas) / len(areas):.2f}" if areas else "  无有效面积数据")

    print("\n各类别出现的图像统计:")
    for cat_id, images in category_image_count.items():
        print(f"{category_mapping.get(cat_id, '未知类别')} ({cat_id}): {len(images)} 张图像")

    """
    柱状图，每个类别实例数的
    """
    # category_count_no_ignore = {cat_id: count for cat_id, count in category_count.items() if count != 9}
    # category_names = [category_mapping.get(cat_id, '未知类别') for cat_id in category_count_no_ignore.keys()]
    # instance_counts = [count for count in category_count_no_ignore.values()]
    #
    # df = pd.DataFrame({
    # "Category": category_names,
    # "#Instances": instance_counts
    # }).sort_values(by="#Instances", ascending=False)
    #
    # cmap = LinearSegmentedColormap.from_list("custom_gradient",["#ADD8E6", "#00008B"], N=len(df))
    # colors=[cmap(i) for i in range(len(df))]
    #
    # plt.figure(figsize=(10,6))
    # ax = sns.barplot(x="Category", y="#Instances", data=df, palette=colors, zorder=2)
    # x_positions = range(len(df))  # 每个类别的位置
    # plt.plot(x_positions, df["#Instances"], marker="o", color="black", linestyle="-", linewidth=1, zorder=3)
    # for i, v in enumerate(df["#Instances"]):
    #     ax.text(i, v + 50, f'{v}', ha='center', va='bottom', fontsize=10, color='black')
    #
    # plt.title("#Instances On Each Category",fontsize=16)
    # plt.xlabel("Category", fontsize=12)
    # plt.ylabel("#Instances", fontsize=12)
    # plt.xticks(rotation=45, ha="right")
    # plt.tight_layout()
    #
    # plt.savefig("instance_number.png", dpi=400, bbox_inches="tight")
    # plt.show()

    """
    每个类别在多少张图像上出现过的实例数
    """
    category_image_count_no_ignore = {cat_id: len(images) for cat_id, images in category_image_count.items() if cat_id != 9}
    category_image_names = [category_mapping.get(cat_id, '未知类别') for cat_id in category_image_count_no_ignore.keys()]
    instance_image_counts = [count for count in category_image_count_no_ignore.values()]
    print(category_image_names)
    print(instance_image_counts)

    df = pd.DataFrame({
        "Category": category_image_names,
        "#Images": instance_image_counts
    }).sort_values(by="#Images", ascending=False)

    cmap = LinearSegmentedColormap.from_list("custom_gradient", ["#ADD8E6", "#00008B"], N=len(df))
    colors = [cmap(i) for i in range(len(df))]

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Category", y="#Images", data=df, palette=colors, zorder=2)
    x_positions = range(len(df))  # 每个类别的位置
    plt.plot(x_positions, df["#Images"], marker="o", color="black", linestyle="-", linewidth=1, zorder=3)
    for i, v in enumerate(df["#Images"]):
        ax.text(i, v + 50, f'{v}', ha='center', va='bottom', fontsize=10, color='black')

    plt.title("#Images Meets Each Category", fontsize=16)
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("#Images", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig("image_number.png", dpi=400, bbox_inches="tight")
    plt.show()