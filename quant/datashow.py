# 导入sns模块
from opendatatools import sns
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

import matplotlib.pyplot as plt
import datetime


def plot_trend(keywords):
    plt.figure(figsize=(14, 8))

    for keyword in keywords:
        df, msg = sns.get_weibo_index(keyword, '3month')

        time = [datetime.datetime.strptime(x, "%Y%m%d") for x in df.index]
        values = [float(x) for x in df[df.columns[0]]]
        plt.plot(time, values, label=keyword)

    plt.xlabel("日期")
    plt.ylabel("热度")
    plt.title('微博指数')
    plt.legend()
    plt.show()

plot_trend(['贸易战'])
