# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 21:48:49 2020

@author: User
"""
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
# nltk.download('stopwords')
# nltk.download('wordnet')

from dbconnect import *
from test_set import *
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
import numpy as np
from wordcloud import WordCloud

db=dbconnect()
train_review=db.all_review()    

db_test=test_set()
test_review=db_test.all_review()

all_review=train_review.append(test_review,ignore_index=True)

all_review = all_review[pd.notnull(all_review['review'])]
print(all_review.shape)




fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
xaxis=all_review['senti_category'].value_counts().index.tolist()
yaxis=all_review['senti_category'].value_counts().tolist()

ax.bar(xaxis,yaxis)

documents = []

stemmer = WordNetLemmatizer()

for sen in all_review['review']:
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(sen))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = sent_tokenize(document)

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)

vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7,
                             stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents)

# frequency count
sum_words = X.sum(axis=0) 
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)


x_val = [x[0] for x in words_freq]
y_val = [x[1] for x in words_freq]

visual=pd.DataFrame(words_freq,columns=['words','freq'])

plt.figure(figsize=(15,10))
visual[:20].plot.bar(x='words',y='freq')
plt.xticks(rotation=50)
plt.xlabel("Words")
plt.ylabel("Counts")
plt.show()




wordcloud = WordCloud(width = 800, height = 800, background_color ='white')
wordcloud.generate_from_frequencies(frequencies=dict(words_freq))
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


features=vectorizer.get_feature_names()    


tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

X_df=pd.DataFrame(X,columns=features)


y=all_review['senti_category']

X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=0, shuffle=True)



models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]

CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
    
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
std_accuracy = cv_df.groupby('model_name').accuracy.std()

acc = pd.concat([mean_accuracy, std_accuracy], axis= 1, 
          ignore_index=True)
acc.columns = ['Mean Accuracy', 'Standard deviation']
print(acc)

plt.figure(figsize=(8,5))
sns.boxplot(x='model_name', y='accuracy', data=cv_df, color='lightblue', showmeans=True)
plt.title("MEAN ACCURACY (cv = 5)\n", size=14);

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


data=confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


df_cm = pd.DataFrame(data, columns=np.unique(y_pred),
                     index = np.unique(y_pred))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="RdBu", annot=True,annot_kws={"size": 16},fmt='d') # font size

