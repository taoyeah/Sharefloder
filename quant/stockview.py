import tushare as ts
import pandas as pd
from opendatatools import stock
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

mystock = \
    '603063', \
    '000993', \
    '002797', \
    '515000', \
    '601519'
df = ts.get_realtime_quotes(mystock) #Single stock symbol
print(df)

df = ts.get_tick_data('603063', date='2019-09-17',src='tt')

# df, msg = stock.get_daily('002008.SZ', start_date='2018-06-06', end_date='2019-09-17')
# df, msg = stock.get_realtime_money_flow('603063.SH')
# # 获取实时行情
# df, msg = stock.get_quote('603063.SH')
print(df)
# print(msg)

x = df['time']
y1 = df['price']
y2 = df['volume']

# plt.figure()
# plt.ylim((-2, 15))
# ax1.plot(x, y1)
# ax2 = ax1.
# plt.plot(x, y2, color='red')
# plt.show()

fig, ax1 = plt.subplots(figsize=(12,6))
plot1 = ax1.plot(x, y1, color='red')
# ax1.set_ylim(0, 24000)

ax2 = ax1.twinx()
plot2 = ax2.plot(x, y2)
# ax2.set_ylim(0, 15)

lines = plot1 + plot2
ax1.legend(lines, ['price', 'volume'])

plt.show()


# df.to_csv('data.csv')
# show = df[['code','name','open','price','volume']]
# print(show)