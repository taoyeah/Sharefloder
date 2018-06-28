"""
To know more or get code samples, please visit my website:
https://morvanzhou.github.io/tutorials/
Or search: 莫烦Python
Thank you for supporting!
"""

# please note, all tutorial code are running under python3.5.
# If you use the version like python2.7, please modify the code accordingly

# 4 - Regressor example

import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

# 读取数据
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
submit = pd.read_csv("data/sample_submit.csv")

# 删除id
train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

# 取出训练集的y
target = train.pop('y')

xx=train.values
zz=test.values
yy=np.array(target)
yyy=yy.reshape(yy.__len__(),1)

# xx = preprocessing.scale(xx)    # normalization step

X_train, X_test, y_train, y_test = train_test_split(
    xx, yyy, test_size=0.4)


# build a neural network from the 1st layer to the last layer
model = Sequential()

model.add(Dense(units=100, input_dim=7, activation='sigmoid'))

model.add(Dense(100, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# choose loss function and optimizing method
model.compile(loss='mse', optimizer='Adam')

# training
print('Training -----------')
for step in range(5001):
    cost = model.train_on_batch(X_train, y_train)
    if step % 50 == 0:
        loss = np.sqrt(model.evaluate(X_test, y_test))
        print('train loss: ', loss)

# # test
# print('\nTesting ------------')
# cost = model.evaluate(X_test, y_test, batch_size=40)
# print('test cost:', cost)
# W, b = model.layers[0].get_weights()
# print('Weights=', W, '\nbiases=', b)
#
# # plotting the prediction
# Y_pred = model.predict(zz)
# print(Y_pred)
