
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt

Train_network = False


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # here to dropout
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs



# 读取数据
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
submit = pd.read_csv("data/sample_submit.csv")

# 删除id
train.drop('id', axis=1, inplace=True)

train.drop('position', axis=1, inplace=True)
train.drop('ground_floor_type', axis=1, inplace=True)
train.drop('roof_type', axis=1, inplace=True)
train.drop('foundation_type', axis=1, inplace=True)
train.drop('land_condition', axis=1, inplace=True)

test.drop('position', axis=1, inplace=True)
test.drop('ground_floor_type', axis=1, inplace=True)
test.drop('roof_type', axis=1, inplace=True)
test.drop('foundation_type', axis=1, inplace=True)
test.drop('land_condition', axis=1, inplace=True)

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

# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 8])
ys = tf.placeholder(tf.float32, [None, 1])
# add hidden layer
l1 = add_layer(xs, 8, 50, 'l1', activation_function=tf.nn.sigmoid)
# # # # add hidden layer
# l2 = add_layer(l1, 200, 60, 'l2',  activation_function=tf.nn.sigmoid)
# # # add hidden layer
# l3 = add_layer(l2, 60, 30, activation_function=tf.nn.sigmoid)
# # # add hidden layer
# l4 = add_layer(l3, 60, 60, activation_function=tf.nn.relu)
# # add output layer
prediction = add_layer(l1, 50, 1, 'l-pre', activation_function=None)

# the error between prediction and real data
loss = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1])))

tf.summary.scalar('loss', loss)

train_step = tf.train.AdamOptimizer(0.2).minimize(loss)
# important step
sess = tf.Session()
merged = tf.summary.merge_all()
# summary writer goes in here
train_writer = tf.summary.FileWriter("logs/train2", sess.graph)
test_writer = tf.summary.FileWriter("logs/test2", sess.graph)

init = tf.global_variables_initializer()

saver = tf.train.Saver()


if Train_network:
    sess.run(init)
else:
    saver.restore(sess, "my_net/save_net.ckpt")

print(X_train)
print(yyy)

mylossb = 100
if Train_network:
    for i in range(600):
        # training
        sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
        if i % 50 == 0:
            train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
            test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
            train_writer.add_summary(train_result, i)
            test_writer.add_summary(test_result, i)
            # to see the step improvement
            myloss = sess.run(loss, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
            if myloss<mylossb:
                mylossb = myloss
            print(myloss)
            print(i)
else:
    myloss = sess.run(loss, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
    print(myloss)

if Train_network:
    saver.save(sess, "my_net/save_net.ckpt")

y_pred = sess.run(prediction, feed_dict={xs: zz, keep_prob: 1})

for i in range(test.__len__()):
    if y_pred[i]<0:
        y_pred[i]=0

submit['y'] = y_pred
submit.to_csv('taoyeah-net5.csv', index=False)
