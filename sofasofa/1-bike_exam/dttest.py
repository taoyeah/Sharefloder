# -*- coding: utf-8 -*-

# 引入模块
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
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


xx = preprocessing.scale(train)    # normalization step

X_train, X_test, y_train, y_test = train_test_split(
    train, y_train, test_size=0.2)

# 建立最大深度为5的决策树回归模型
reg = DecisionTreeRegressor(max_depth=10)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)


yy=np.array(y_test)
yyy=yy.reshape(yy.__len__(), 1)


yyp=np.array(y_pred)
yyyp=yyp.reshape(yyp.__len__(), 1)


#loss = tf.reduce_mean(tf.square(yyy-yyyp))
loss = tf.reduce_mean(tf.reduce_sum(tf.square(yyy-yyyp), reduction_indices=[1]))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(loss))

print(reg.score(X_test,y_test))

# 输出预测结果至my_DT_prediction.csv
# submit['y'] = y_pred
# submit.to_csv('taoyeah-DT-prediction.csv', index=False)