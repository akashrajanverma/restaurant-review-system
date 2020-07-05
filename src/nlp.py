#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 03:13:15 2019

@author: arv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the .tsv file
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

#cleaning the dataset
import re
import nltk #import natural language toolkit
nltk.download('stopwords')
from nltk.corpus import stopwords
#convert all the upper case to lower case
from nltk.stem.porter import PorterStemmer


corpus = []   #contains all the words of each review
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    ps = PorterStemmer()
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#creatting a bag of words
#all the different words in a column of a sparse matrix
#all the reviews in the rows
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[: , 1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state = 0)

#fitting Naive Bayes

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

#predicting the X_test result

y_pred = classifier.predict(X_test)

#making confusing matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
