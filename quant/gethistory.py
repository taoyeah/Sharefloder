import tushare as ts
pro = ts.pro_api('454a33eed89b1b5cf7a08145207e9847fcb41791e4afa09cf6714931')
df = pro.trade_cal(exchange='', start_date='20180901', end_date='20181001', fields='exchange,cal_date,is_open,pretrade_date', is_open='0')
print(df)
