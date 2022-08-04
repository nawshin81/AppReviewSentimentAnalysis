# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 20:26:54 2020

@author: Raida
"""
from dbconnect import *

class test_set:
    def __init__(self):
        database=dbconnect()
        self.cur=database.cur
        
        
    def test_bug(self):
        bug=[]
        self.cur.execute("SELECT * FROM bug_report_data_test" )
        for i in self.cur.fetchall():
            if i[13] >0:
                s='positive'
            elif i[13]<0:
                s='negative'
            else:
                s='neutral'
            bug.append((i[4],i[14],i[15],s,'Bug_Report'))
        return bug
      
    def test_feature_request(self):
        feature_request=[]
        self.cur.execute("SELECT * FROM feature_or_improvment_request_data_test" )
        for i in self.cur.fetchall():
            if i[13] >0:
                s='positive'
            elif i[13]<0:
                s='negative'
            else:
                s='neutral'
            feature_request.append((i[4],i[14],i[15],s,'Feature_Request'))
        return feature_request
    
    def test_user_experience(self):
        user_experience=[]
        self.cur.execute("SELECT * FROM userexperience_data_test" )
        for i in self.cur.fetchall():
            if i[13] >0:
                s='positive'
            elif i[13]<0:
                s='negative'
            else:
                s='neutral'
            user_experience.append((i[4],i[14],i[15],s,'User_Experience'))
        return user_experience
    
    def test_rating(self):
        rating=[]
        self.cur.execute("SELECT * FROM rating_data_test" )
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
        bug=self.test_bug()
        feature_request=self.test_feature_request()
        user_experience=self.test_user_experience()
        rating=self.test_rating()
        review=bug+feature_request+user_experience+rating
        self.review_df_test=pd.DataFrame(review,columns=['review','senti_pos','senti_neg','senti_category','category'])
        return self.review_df_test
    
# if __name__=='__main__':
#     test=test_set()
#     print(test.all_review())

