from opendatatools import stock
from opendatatools import sns
df_stock,msg = stock.get_daily('603063.SH', '2019-04-16', '2019-05-17')
df_weibo_index, msg = sns.get_weibo_index('风电', '1month')

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

import matplotlib.pyplot as plt
import datetime

plt.figure(figsize=(14, 8))

ax = plt.subplot(1,1,1)

time   = [x for x in df_stock['time']]
values = [float(x) for x in df_stock['percent'] ]
ax.plot(time, values, label='股价涨跌', color='blue')
ax.legend()

ax2 = ax.twinx()
time   = [datetime.datetime.strptime(x, "%Y%m%d") for x in df_weibo_index.index]
values = [float(x) for x in df_weibo_index[df_weibo_index.columns[0]]]
ax2.plot(time, values, label='微博热度', color='red')
ax2.legend()

plt.xlabel("日期")
plt.ylabel("change")
plt.title('股价和热度对比')
plt.legend()
plt.show()