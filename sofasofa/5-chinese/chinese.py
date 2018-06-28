# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from gensim.models import Word2Vec

train = pd.read_csv('data/train.txt')
test = pd.read_csv('data/test.txt')
submit = pd.read_csv('data/sample_submit.csv')

total = len(train) + len(test)
n_train = len(train)

labeled_texts = []

texts = list(train['text']) + list(test['text'])

ndims = 100
model = Word2Vec(sentences=texts, size=ndims)

vecs = np.zeros([total, ndims])
for i, sentence in enumerate(texts):
    counts, row = 0, 0
    for char in sentence:
        try:
            if char != ' ':
                row += model.wv[char]
                counts += 1
        except:
            pass
    if counts == 0:
        print(sentence)
    vecs[i, :] = row / counts

clf = DecisionTreeClassifier(max_depth=3, random_state=50)
clf.fit(vecs[:n_train], train['y'])
print(clf.score(vecs[:n_train], train['y']))
submit['y'] = clf.predict_proba(vecs[n_train:])[:, 1]
submit.to_csv('chinese1.csv', index=False)
