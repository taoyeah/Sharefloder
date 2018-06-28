# -*- coding: utf-8 -*-

# 引入模块
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
import pandas as pd

# 读取数据
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
submit = pd.read_csv("data/sample_submit.csv")

# 删除id
train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

# 取出训练集的y
y_train = train.pop('y')

# 建立最大深度为5的决策树回归模型
reg = DecisionTreeRegressor(max_depth=10)
reg.fit(train, y_train)
y_pred = reg.predict(test)

# 输出预测结果至my_DT_prediction.csv
submit['y'] = y_pred
submit.to_csv('taoyeah-DT-prediction.csv', index=False)