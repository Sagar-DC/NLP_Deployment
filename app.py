from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def text_clean(msg):

    stemmer = PorterStemmer()
    
    for i in range(0, len([msg])):
        text = re.sub('[^a-zA-Z]', ' ', msg)
        text = text.lower()
        text = text.split()
    
        text = [stemmer.stem(word) for word in text if not word in stopwords.words('english')]
        text = ' '.join(text)
    
    return text


# load the model from disk
filename = 'model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb')) 
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        #message = text_clean(message)
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('index.html',prediction_text = my_prediction, cleaned_text = vect.shape)

@app.route('/clean',methods=['POST'])
def clean():
    if request.method == 'POST':
        message = request.form['message']
        for i in range(len(message)):
            text = re.sub('[^a-zA-Z]', ' ', message)
            
        message = text_clean(message)
        message = message
    return render_template('index.html',cleaned_text = "Stemmed message : {}".format(text), actual_text = "actual message : {}".format(message))


if __name__ == '__main__':
	app.run(debug=True)