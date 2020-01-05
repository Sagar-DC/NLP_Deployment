from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from Data_Preprocessor import text_clean
import pickle

# load the model from disk
filename = 'model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))
cl=pickle.load(open('clean.pkl','rb')) 
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = cl(message)
        data = [data]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)