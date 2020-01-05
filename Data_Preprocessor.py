# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 20:40:15 2019

@author: sagar
"""
import pickle

def text_clean(msg):
    import re
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer

    stemmer = PorterStemmer()
    
    for i in range(0, len([msg])):
        text = re.sub('[^a-zA-Z]', ' ', msg)
        text = text.lower()
        text = text.split()
    
        text = [stemmer.stem(word) for word in text if not word in stopwords.words('english')]
        text = ' '.join(text)
    
    return text


pickle.dump(text_clean, open('clean.pkl', 'wb'))


