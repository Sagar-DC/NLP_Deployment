from flask import Flask,render_template,url_for,request
import pandas as pd 
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

# load the model from disk
filename = 'model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))  
#cl=pickle.load(open('clean.pkl','rb')) 
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = text_clean(request.form['message'])
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('index.html',prediction_text = my_prediction)


@app.route('/clean',methods=['POST'])
def clean():
	if request.method == 'POST':
		message = request.form['message']
		text = text_clean(message)
	return render_template('index.html',cleaned_text = "Cleaned Message: {}".format(text), actual_text = "Message Entered: {}".format(message)) 

if __name__ == '__main__':
	app.run(debug=True)
