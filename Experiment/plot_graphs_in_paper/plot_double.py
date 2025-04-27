"""
绘制折线图, 一个x两个y的时候可以使用
参考链接：https://blog.csdn.net/weixin_38037405/article/details/124719333
"""
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_double_y(x, y1, y2, color_y1, color_y2, title, xlabel, ylabel, save_path):
    """
    绘制折线图，两个数据进行比较的，高的会比低的有颜色填充
    Args:
        x: 横轴
        y1: 第一个数据
        y2: 第二个数据
        color_y1: 第一个数据的颜色
        color_y2: 第二个数据的颜色
        title: 表头的题目
        xlabel: x轴的标签
        ylabel: y轴的标签
        save_path: 保存的路径

    Returns: None
    """
    # ---- 读取数据
    x = x
    y1 = y1
    y2 = y2

    # ---- 临时变量，为图像中填充所用
    x_aux = x.copy()
    x_aux.index = x_aux.index * 10
    last_idx = x_aux.index[-1] + 1
    x_aux = x_aux.reindex(range(last_idx))
    x_aux = x_aux.interpolate()

    y1_aux = y1.copy()
    y1_aux.index = y1_aux.index * 10
    last_idx = y1_aux.index[-1] + 1
    y1_aux = y1_aux.reindex(range(last_idx))
    y1_aux = y1_aux.interpolate()

    y2_aux = y2.copy()
    y2_aux.index = y2_aux.index * 10
    last_idx = y2_aux.index[-1] + 1
    y2_aux = y2_aux.reindex(range(last_idx))
    y2_aux = y2_aux.interpolate()

    # plt绘图全局字体设置
    mpl.rcParams["font.sans-serif"] = ["Times New Roman"]

    # ---- 绘图
    fig, ax = plt.subplots(figsize=(13, 5))
    (line1,) = ax.plot(
        x,
        y1,
        label=data.columns[2],
        marker="o",
        mfc="white",
        ms=8,
        color=color_y1,
        linewidth=4,
    )
    (line2,) = ax.plot(
        x,
        y2,
        label=data.columns[3],
        marker="s",
        mfc="white",
        ms=8,
        color=color_y2,
        linewidth=4,
    )
    # * mfc：设置中间为空白，白色；ms：控制marker的大小

    # ---- 设置背景网络的形式
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(True, linestyle="--", color="#4E616C", alpha=0.6)

    # ---- 填充颜色
    for i in range(len(x_aux) - 1):
        if y1_aux.iloc[i + 1] > y2_aux[i + 1]:
            color = line1.get_color()
        else:
            color = line2.get_color()

        plt.fill_between(
            [x_aux[i], x_aux[i + 1]],
            [y1_aux.iloc[i], y1_aux.iloc[i + 1]],
            [y2_aux.iloc[i], y2_aux.iloc[i + 1]],
            color=color,
            zorder=2,
            alpha=0.2,
            ec=None,
        )

    # ax.set_title(title, fontsize=18, pad=18)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)

    # ---- 图例设置
    y1_last = y1.iloc[-1]
    y2_last = y2.iloc[-1]

    y1_lengend = ax.text(
        x=x.iloc[-1] + 2,
        y=y1_last,
        s=f"{y1_last:,.1f}, {data.columns[2]}",
        color=line1.get_color(),
        va="center",
        ha="left",
        size=18,
    )

    y2_lengend = ax.text(
        x=x.iloc[-1] + 2,
        y=y2_last,
        s=f"{y2_last:,.1f}, {data.columns[3]}",
        color=line2.get_color(),
        va="center",
        ha="left",
        size=18,
    )

    # ---- 设置坐标轴与刻度
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_tick_params(
        length=2, color="#4E616C", labelcolor="#4E616C", labelsize=14
    )
    ax.yaxis.set_tick_params(
        length=2, color="#4E616C", labelcolor="#4E616C", labelsize=14
    )
    ax.spines["bottom"].set_edgecolor("#4E616C")

    plt.tight_layout()
    plt.show()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    excel_path = (
        "/mnt/d/learning/空天院/论文/01-paper1/SCA计算/visdrone/similar_object_statistics/结果统计.xlsx"
    )
    data = pd.read_excel(excel_path)
    save_path = "/mnt/d/learning/空天院/论文/01-paper1/Python绘图/不同目标数量的SCA变化—visdrone-155.svg"

    x = data.iloc[:, 0]
    y1 = data.iloc[:, 2]
    y2 = data.iloc[:, 3]
    title = "SCA on different #Objects"
    xlabel = data.columns[0]
    ylabel = "aSCA"
    plot_double_y(
        x=x,
        y1=y1,
        y2=y2,
        color_y1="#F2AE66",
        color_y2="#C30E59",
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        save_path=save_path,
    )
