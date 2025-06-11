"""
散点图-回归图-生成过程
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns
import matplotlib as mpl

sns.despine(top=True, right=True)
sns.set_style("white")
sns.set_context("notebook", font_scale=1, rc={"figure.figsize": (12, 8)})
mpl.rcParams["font.sans-serif"] = ["Times New Roman"]


def data_get(scores_path, idx_path, category_x, category_y):
    # ---- 读取数据
    scores_pd = pd.read_csv(scores_path)
    idx_pd = pd.read_csv(idx_path)

    # ---- 检查是否存在脏数据，清楚掉重复数据、确保每个索引是整数，只保留有效索引
    scores_pd = scores_pd.iloc[:, 1:]
    idx_values = idx_pd.iloc[:, 1].drop_duplicates().values
    scores_pd.index = scores_pd.index.astype(int)  # 确保 `scores_pd` 的索引是整数
    idx_values = [int(i) for i in idx_values]  # 确保 `idx_values` 也是整数
    valid_idx = scores_pd.index.intersection(idx_values)  # 只保留有效索引
    result = scores_pd.loc[valid_idx - 2]

    # ---- 得到进行计算的数据，并按照大小赋以不同的标签
    # similar = result.iloc[:, [3, 5]].copy()
    similar = result.iloc[:, [2, 7]].copy()
    similar["label"] = similar.apply(
        lambda row: category_x if row.iloc[0] > row.iloc[1] else category_y, axis=1
    )
    similar_x = similar[similar["label"] == category_x]
    similar_y = similar[similar["label"] == category_y]

    # ---- 对两类散点分别进行线性回归
    slope_x, intercept_x, _, _, _ = linregress(
        similar_x.iloc[:, 0], similar_x.iloc[:, 1]
    )
    slope_y, intercept_y, _, _, _ = linregress(
        similar_y.iloc[:, 0], similar_y.iloc[:, 1]
    )

    # ---- 求出回归直线之间的角度
    angle = np.arctan(np.abs((slope_y - slope_x) / (1 + slope_y * slope_x)))
    angle_degrees = np.degrees(angle)
    return similar, angle_degrees


def plot_similar_category_vis(
    data, angle, category_x: str, category_y: str, color: list, save_path
):
    # ---- 散点图不同类别的颜色设置
    color = color

    # ---- 创建画布与子图布局
    # fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    # plt.tight_layout(pad=3)

    # ---- 公共的参数进行设置
    x_col = data.columns[0]
    y_col = data.columns[1]

    # ---- 第一个图：不分类别的散点图
    plt.figure(figsize=(6,5))
    sns.scatterplot(
        x=x_col,
        y=y_col,
        data=data,
        color="#27374D",
        # ax=axes[0],
        s=15,
        edgecolor="white",
        label="All Data",
    )
    # axes[0].set(xlabel=category_x, ylabel=category_y)
    # axes[0].xaxis.set_major_locator(plt.MultipleLocator(0.1))
    # axes[0].yaxis.set_major_locator(plt.MultipleLocator(0.1))
    # axes[0].set_xlim(0, 1)
    # axes[0].set_ylim(0, 1)
    # axes[0].spines['top'].set_visible(False)
    # axes[0].spines['right'].set_visible(False)
    # axes[0].legend()
    plt.xlabel(category_x, fontsize=15)
    plt.ylabel(category_y, fontsize=15)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.1))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(prop={'size':15})
    plt.savefig(save_folder_path + "all_data_scatter.tif", dpi=600, bbox_inches='tight')

    # ---- 第二个图：按照不同类别为散点图上的点赋以不同的颜色
    plt.figure(figsize=(6,5))
    sns.scatterplot(
        x=x_col,
        y=y_col,
        hue="label",
        hue_order=[category_x, category_y],
        palette=color,
        data=data,
        # ax=axes[1],
        s=15,
        edgecolor="white",
    )
    # axes[1].set(xlabel=category_x, ylabel=category_y)
    # axes[1].xaxis.set_major_locator(plt.MultipleLocator(0.1))
    # axes[1].yaxis.set_major_locator(plt.MultipleLocator(0.1))
    # axes[1].set_xlim(0, 1)
    # axes[1].set_ylim(0, 1)
    # axes[1].spines['top'].set_visible(False)
    # axes[1].spines['right'].set_visible(False)
    # axes[1].legend(title=None)  # 去掉图例的标题
    plt.xlabel(category_x, fontsize=15)
    plt.ylabel(category_y, fontsize=15)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.1))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(prop={'size':15})
    plt.savefig(save_folder_path + "category_scatter.tif", dpi=600, bbox_inches='tight')

    # ---- 第三个图：为不同的类别的图画上回归直线
    plt.figure(figsize=(6,5))
    for idx, category in enumerate([category_x, category_y]):
        subset = data[data["label"] == category]
        sns.regplot(
            x=x_col,
            y=y_col,
            data=subset,
            color=color[idx],
            truncate=True,
            # ax=axes[2],
            scatter_kws={"s": 15, "edgecolor": "white"},
            label=category,
        )
    # axes[2].set(xlabel=category_x, ylabel=category_y)
    # axes[2].xaxis.set_major_locator(plt.MultipleLocator(0.1))
    # axes[2].yaxis.set_major_locator(plt.MultipleLocator(0.1))
    # axes[2].set_xlim(0, 1)
    # axes[2].set_ylim(0, 1)
    # axes[2].spines['top'].set_visible(False)
    # axes[2].spines['right'].set_visible(False)
    # axes[2].legend()
    plt.xlabel(category_x, fontsize=18)
    plt.ylabel(category_y, fontsize=18)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.2))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.2))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(prop={'size':18})
    plt.savefig(save_folder_path + "category_regression.tif", dpi=600, bbox_inches='tight')

    # ---- 第四个图：带上了回归直线之间夹角的图
    # for idx, category in enumerate([category_x, category_y]):
    #     subset = data[data["label"] == category]
    #     sns.regplot(
    #         x=x_col,
    #         y=y_col,
    #         data=subset,
    #         color=color[idx],
    #         truncate=True,
    #         ax=axes[3],
    #         scatter_kws={"s": 15, "edgecolor": "white"},
    #         label=category,
    #     )
    # axes[3].set(xlabel=category_x, ylabel=category_y)
    # axes[3].xaxis.set_major_locator(plt.MultipleLocator(0.1))
    # axes[3].yaxis.set_major_locator(plt.MultipleLocator(0.1))
    # axes[3].set_xlim(0, 1)
    # axes[3].set_ylim(0, 1)
    # axes[3].spines['top'].set_visible(False)
    # axes[3].spines['right'].set_visible(False)
    # plt.legend()

    # ---- 打印该图片
    # plt.savefig(save_path, dpi=400, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    scores_path = "/mnt/d/learning/空天院/论文/01-paper1/0-具有相似特征的目标（large-vehicle和container）/162_800_1300_650_different_category_vis/s2anet_refine_scores_scale_1.csv"
    # scores_path = "/mnt/d/exp/sodaa_sob/paper1_a6000result/0-comparison_exp/0924_baseline/test/vis_paper_0526/196/196_scores_1.csv"
    idx_path = "/mnt/d/learning/空天院/论文/01-paper1/0-具有相似特征的目标（large-vehicle和container）/162_800_1300_650_different_category_vis/s2anet_refine_idx_scale_1.csv"
    # idx_path = "/mnt/d/exp/sodaa_sob/paper1_a6000result/0-comparison_exp/0924_baseline/test/vis_paper_0526/196/196_idx_1.csv"
    # category_x = "large vehicle"
    category_x = "small vehicle"
    # category_y = "container"
    category_y = "swimming pool"
    save_folder_path = "/mnt/d/learning/空天院/论文/01-paper1/visio画图/1-修改图/Introducition里的介绍图/162_vis/"
    # save_folder_path = "/mnt/d/exp/sodaa_sob/paper1_a6000result/0-comparison_exp/0924_baseline/test/vis_paper_0526/196/vis_new"

    data, angle = data_get(
        scores_path=scores_path,
        idx_path=idx_path,
        category_x=category_x,
        category_y=category_y,
    )
    print(angle)
    color = ["#2673EF", "#EC6B2D"]
    plot_similar_category_vis(
        data=data,
        angle=angle,
        category_x=category_x,
        category_y=category_y,
        color=color,
        save_path=save_folder_path,
    )
