# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 18:34:55 2019

@author: sagar
"""

from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from Data_Preprocessor import text_clean

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('tranform.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        text = text_clean(message)
        text = [text]
        vector = cv.transform(text).toarray()
        prediction = model.predict(vector)
    return render_template('index.html', prediction_text = prediction)

@app.route('/clean', methods = ['POST'])
def clean():
    message = request.form['message']
    text = text_clean(message)
    return render_template('index.html', cleaned_text = "Cleaned Message: {}".format(text))

if __name__ == "__main__":
    app.run(debug = True)
