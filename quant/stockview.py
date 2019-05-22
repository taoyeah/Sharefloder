import tushare as ts
import pandas as pd
mystock = \
    '603063', \
    '000993', \
    '002797', \
    '002008', \
    '601519'
df = ts.get_realtime_quotes(mystock) #Single stock symbol
show = df[['code','name','open','price','volume']]
print(show)