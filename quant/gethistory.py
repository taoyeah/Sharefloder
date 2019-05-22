import tushare as ts
pro = ts.pro_api('454a33eed89b1b5cf7a08145207e9847fcb41791e4afa09cf6714931')
df = pro.daily(ts_code='000001.SZ', start_date='20180701', end_date='20180718')
print(df)
