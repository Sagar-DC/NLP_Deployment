# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 23:22:49 2019

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
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    msg = request.form['message']
    text = text_clean(msg)
    
    text = [text]
    vector = cv.transform(text).toarray()
    prediction = model.predict(vector)
    
    if prediction == 0:
        output = "Ham"
    else:
        output = "Spam"
    
    return render_template('index.html', prediction_text = prediction)
    #return render_template('index.html', prediction_text='Message Entered is "{}" '.format(output))

@app.route('/clean', methods=['POST'])
def clean():
    msg = request.form['message']
    text = text_clean(msg)
    return render_template('index.html', cleaned_text='Cleaned Message "{}" '.format(text))
    

if __name__ == "__main__":
    app.run(debug=True)