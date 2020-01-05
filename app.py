from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# load the model from disk
filename = 'model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb')) 

stemmer = PorterStemmer()
stop = stopwords.words('english')
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

        text = re.sub('[^a-zA-Z]', ' ', message)
        text = text.lower()
        text = text.split()
        for word in text:
            #if not word in stopwords.words('english'):
                text = stemmer.stem(word) 


        #text = [stemmer.stem(word) for word in text if not word in stopwords.words('english')]
        #text = ' '.join(text)
            
    return render_template('index.html',cleaned_text = "Stemmed message : {}".format(text), actual_text = "Actual message : {}".format(message))


if __name__ == '__main__':
    app.run(debug=True)