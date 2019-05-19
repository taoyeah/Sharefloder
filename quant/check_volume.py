import tushare as ts
import pandas as pd
dg = ts.get_realtime_quotes('601519') #Single stock symbol
show1=dg[['code','name','price','bid','ask','volume','amount','time']]

df = ts.get_sina_dd('601519', date='2019-05-17', vol=900)
buy = df[df['type']=='买盘']
sale = df[df['type']=='卖盘']
buy_amount = buy.volume*buy.price
buy_average_price = buy_amount.sum()/buy.volume.sum()

sale_amount = sale.volume*sale.price
sale_average_price = sale_amount.sum()/sale.volume.sum()

show = pd.DataFrame(columns=['buy','sale'],data=[[buy_average_price,sale_average_price],[buy.volume.sum(),sale.volume.sum()]])

print(show)
print(show1)

# print(buy_average_price)
# print(buy.volume.sum())
#
#
# print(sale_average_price)
# print(sale.volume.sum())