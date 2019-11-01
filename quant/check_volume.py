import tushare as ts
import matplotlib.pyplot as plt
import pandas as pd
mystock = \
    '603063.SH,' \
    '000993.SZ,' \
    '002797.SH,' \
    '002008.SZ,' \
    '601519.SH'
dg = ts.get_realtime_quotes('515000') #Single stock symbol
show1=dg[['code','name','price','bid','ask','volume','amount','time']]

df = ts.get_tick_data('601519', date='2019-09-12',src='tt')
# df = ts.get_hist_data('601519', '2019-09-12','2019-09-16')
# df = ts.get_today_ticks('603063')

buy = df[(df['type']=='买盘')&(df['volume']>100)]
sale = df[(df['type']=='卖盘')&(df['volume']>100)]

buy_amount = buy.volume*buy.price
buy_average_price = buy_amount.sum()/buy.volume.sum()

sale_amount = sale.volume*sale.price
sale_average_price = sale_amount.sum()/sale.volume.sum()

show = pd.DataFrame(columns=['buy','sale'],data=[[buy_average_price,sale_average_price],[buy.volume.sum(),sale.volume.sum()]])
print(" ")
print(show)
print(show1)
# print(df['time'])

print(buy_average_price)
print(buy.volume.sum())


print(sale_average_price)
print(sale.volume.sum())