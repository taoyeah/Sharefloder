import tushare as ts
from opendatatools import fund
from opendatatools import stock
import matplotlib.pyplot as plt
import pandas as pd
mystock = \
    '603063', \
    '002008', \
    '002797', \
    '515000', \
    '601519'
# df = ts.get_realtime_quotes(mystock) #Single stock symbol
# show = df[['code','name','open','price','volume']]
# print(show)
# df = ts.get_today_ticks('601519')

# 根据基金公司获取基金列表
df, msg = fund.get_fund_nav('515000')
df.sort_values('date', inplace=True)
print(df)
df.to_csv('data.csv')
x = df['date']
y1 = df['nav1']
y2 = df['nav2']

# plt.figure()
# plt.plot(x, y1)
# plt.show()




# plt.figure(num=3, figsize=(8, 5),)
# plt.plot(x, y1)
# # plot the second curve in this figure with certain parameters
# plt.plot(x, y2, color='red', linewidth=1.0, linestyle='--')
# plt.show()