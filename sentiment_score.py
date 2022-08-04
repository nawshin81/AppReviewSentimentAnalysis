# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 17:02:16 2020

@author: user
"""

from dbconnect import *
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
# nltk.download('vader_lexicon')
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import csv



class senti_score:
    def __init__(self,df_review):
        self.all_review=df_review
        self.analyzer = SentimentIntensityAnalyzer()
        new_word={}
        with open('words - lexicon update.csv') as file:
            reader=csv.DictReader(file)
            for row in reader:
                senti_score=(float(row['neg'])+float(row['pos']))
                key=row['word']
                new_word[key]=senti_score
        self.analyzer.lexicon.update(new_word)
    
    def sentence_preprocessing(self,sentence):

        tokenized_words = word_tokenize(sentence.lower(), "english")
        lemma_words = []
        for word in tokenized_words:
            word = WordNetLemmatizer().lemmatize(word)
            lemma_words.append(word)
        new_sentence=" ".join(lemma_words)
        return new_sentence
    
    def sentence_scoring(self,sentence):
        scores = self.analyzer.polarity_scores(sentence)
        if scores['compound']<=-0.1:
            return "negative"
        elif scores['compound']>0.1:
            return "positive"
        else:
            return "neutral"
        
    def result_set(self):
        all_sentiment=[]
        for review in self.all_review:
            new_sentence=self.sentence_preprocessing(review)
            sentiment=self.sentence_scoring(new_sentence)
            all_sentiment.append(sentiment)
        return all_sentiment
