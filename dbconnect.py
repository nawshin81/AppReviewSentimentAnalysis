# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 20:52:29 2020
"""

import MySQLdb
import pandas as pd

class dbconnect:
    def __init__(self):
        db = MySQLdb.connect(host="127.0.0.1", 
                     user="root", 
                      db="re2020_training_set") 

        db.autocommit(True)
        db.begin()
        cur = db.cursor()
        self.db = db
        self.cur = cur
            
    def bug_review(self):
        bug=[]
        self.cur.execute("SELECT * FROM Bug_Report_Data_Train" )
        for i in self.cur.fetchall():
            if i[13] >0:
                s='positive'
            elif i[13]<0:
                s='negative'
            else:
                s='neutral'
            bug.append((i[4],i[14],i[15],s,'Bug_Report'))
        return bug
    def feature_request_review(self):
        feature_request=[]
        self.cur.execute("SELECT * FROM  feature_or_improvment_request_data_train" )
        for i in self.cur.fetchall():
            if i[13] >0:
                s='positive'
            elif i[13]<0:
                s='negative'
            else:
                s='neutral'
            feature_request.append((i[4],i[14],i[15],s,'Feature_Request'))
        return feature_request
            
    def user_experience_review(self):
        user_experience=[]
        self.cur.execute("SELECT * FROM  userexperience_data_train" )
        for i in self.cur.fetchall():
            if i[13] >0:
                s='positive'
            elif i[13]<0:
                s='negative'
            else:
                s='neutral'
            user_experience.append((i[4],i[14],i[15],s,'User_Experience'))
        return user_experience

    def rating_review(self):
        rating=[]
        self.cur.execute("SELECT * FROM  rating_data_train" )
        for i in self.cur.fetchall():
            if i[13] >0:
                s='positive'
            elif i[13]<0:
                s='negative'
            else:
                s='neutral'
            rating.append((i[4],i[14],i[15],s,'Rating'))
        return rating
            
    def all_review(self):
        bug=self.bug_review()
        feature_request=self.feature_request_review()
        user_experience=self.user_experience_review()
        rating=self.rating_review()
        review=bug+feature_request+user_experience+rating
        self.review_df=pd.DataFrame(review,columns=['review','senti_pos'
                                                    ,'senti_neg','senti_category'
                                                    ,'category'])
        return self.review_df
        
# if __name__=='__main__':
#     database=dbconnect()
#     database.all_review()