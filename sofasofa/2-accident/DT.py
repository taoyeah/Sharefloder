# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

# 读取数据
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
submit = pd.read_csv("data/sample_submit.csv")

# 删除id
train.drop('CaseId', axis=1, inplace=True)
test.drop('CaseId', axis=1, inplace=True)

# 取出训练集的y
target = train.pop('Evaluation')

X_train, X_test, y_train, y_test = train_test_split(
    train, target, test_size=0.3)

# # 建立随机森林模型
# clf = RandomForestClassifier(n_estimators=100, random_state=0)
# for i in range(10):
#     clf.fit(X_train, y_train)

# build network


#  Evaluate
y_predict=clf.predict_proba(X_test)[:,1]
print(average_precision_score(y_test, y_predict))

y_pred = clf.predict_proba(test)[:, 1]

# 输出预测结果至my_RF_prediction.csv
submit['Evaluation'] = y_pred
submit.to_csv('accident1.csv', index=False)
