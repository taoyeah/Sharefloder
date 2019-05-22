import tushare as ts
import pandas as pd
score = 0;

df = ts.get_tick_data('601519', date='2019-05-20',src='tt')
dg = ts.get_hist_data('601519',start='2019-05-21',end='2019-05-21')
buy = df[(df['type']=='买盘')&(df['volume']>100)]
sale = df[(df['type']=='卖盘')&(df['volume']>100)]

buy_amount = buy.volume*buy.price
buy_average_price = buy_amount.sum()/buy.volume.sum()

sale_amount = sale.volume*sale.price
sale_average_price = sale_amount.sum()/sale.volume.sum()

print(dg.high)

if (buy_amount.sum()>sale_amount.sum())&(buy_average_price>sale_average_price):
    if dg.p_change>1:
        score = score +1

print(score)

