# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 08:04:22 2020

@author: User
"""


from sentiment_score import *
from dbconnect import *
from test_set import *
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

db_test=test_set()
db_train=dbconnect()
train = db_train.all_review()
test=db_test.all_review()

X=train.append(test,ignore_index=True)
X_df=X.iloc[:,0]
y=X['senti_category']

senti=senti_score(X_df)
sentiment_score=senti.result_set()

data=confusion_matrix(y,sentiment_score)
report=classification_report(y,sentiment_score)
print(accuracy_score(y,sentiment_score))


df_cm = pd.DataFrame(data, columns=np.unique(sentiment_score),
                     index = np.unique(sentiment_score))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16},fmt='d')# font size


total_pos=y.value_counts().tolist()[0]
total_neg=y.value_counts().tolist()[1]

negative=data[0]
positive=data[2]

index=['negative','neutral','positive']

df = pd.DataFrame({'Actual positive': positive,
                   'Actual negative': negative}, index=index)

ax = df.plot.barh(stacked=True)

ax.set_xlabel('Number of Actual Class')
ax.set_ylabel('Predicted class')





