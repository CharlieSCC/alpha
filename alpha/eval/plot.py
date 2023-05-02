import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
from alpha.eval.backtest import *


def plot_net_value(profit_series_list, labels, fig_size=(20, 5), title='Net Value Curve', x_label='Date', y_label='Net Value', line_width=2):
    """
    画净值曲线函数

    参数：
    profit_series_list: list，包含多个 pd.Series 类型的列表，每个 Series 代表一个净值序列
    fig_size: tuple，画布大小，默认为 (10, 6)
    title: str，图表标题，默认为 'Net Value Curve'
    x_label: str，x轴标签，默认为 'Date'
    y_label: str，y轴标签，默认为 'Net Value'

    返回：
    None
    """
    plt.figure(figsize=fig_size)
    # 遍历每个净值序列，计算其投资组合价值的时间序列数据，然后画出净值曲线
    for i, profit_series in enumerate(profit_series_list):
        net_value_series = profit_series.cumsum()
        plt.plot(net_value_series, label=labels[i], linewidth=line_width, marker="|")

    # 添加图例
    # 设置图表属性
    # plt.title(title)
    # plt.xlabel(x_label)
    # plt.ylabel(y_label)
    plt.legend()
    locator = AutoDateLocator()
    formatter = AutoDateFormatter(locator)
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()


if __name__ == "__main__":
    b = BackTest(1000000, 0.00, 0.00)

    hats = pd.read_csv("/home/chencheng/hats_0.0.1/info_20210101_20221231.csv", index_col=0)
    ret_hats = b.get_daily_pnl(hats, 0.1).groupby("date").sum().shift(1).fillna(0)
    ret_hats.index = pd.to_datetime(ret_hats.index, format="%Y%m%d")

    gru = pd.read_csv("/home/chencheng/gru_0.0.1/info_20210101_20221231.csv", index_col=0)
    ret_gru = b.get_daily_pnl(gru, 0.1).groupby("date").sum().shift(1).fillna(0)
    ret_gru.index = pd.to_datetime(ret_gru.index, format="%Y%m%d")

    zz800 = gru.groupby("date")["ret"].mean().shift(1).fillna(0)
    zz800.index = pd.to_datetime(zz800.index, format="%Y%m%d")

    plot_net_value([ret_hats, ret_gru, zz800], ["HATS", "GRU", "CSI800"])
