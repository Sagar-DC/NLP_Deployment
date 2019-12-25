# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 20:38:24 2019

@author: sagar
"""

import pandas as pd
import numpy as np

data = pd.read_csv("G:\\Deployment(Practice)\\NLP_Deployment\\Data\\ham_spam.csv",encoding = "ISO-8859-1")

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []

for i in range(0, len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data['text'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


from sklearn.feature_extraction.text import CountVectorizer  
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(data['type'])
y = y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(y_test, y_pred)

accuracy_score(y_test, y_pred)

# Saving model to disk
import pickle
pickle.dump(spam_detect_model, open('model.pkl','wb'))

pickle.dump(cv, open('tranform.pkl', 'wb'))
