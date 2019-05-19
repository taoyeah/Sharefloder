import tushare as ts
df = ts.get_sina_dd('000993', date='2019-05-17', vol=1000)
print(df)