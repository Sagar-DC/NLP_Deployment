from flask import Flask,render_template,url_for,request
import pickle

import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords

def stemm_text(msg):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    
    for i in range(0, len([msg])):
        text = re.sub('[^a-zA-Z]', ' ', msg)
        text = text.lower()
        text = text.split()
        
        text = [stemmer.stem(word) for word in text if not word in stopwords.words('english')]
        text = ' '.join(text)
        
    return text

def lemmatize_text(msg):
    from nltk.stem import WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()
    
    for i in range(0, len([msg])):
        text = re.sub('[^a-zA-Z]', ' ', msg)
        text = text.lower()
        text = text.split()
        
        text = [wordnet_lemmatizer.lemmatize(word) for word in text if not word in stopwords.words('english')]
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

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
    else:
        message = request.args.get('message')
    
    message = stemm_text(message)
    data = [message]
    vect = cv.transform(data).toarray()
    my_prediction = clf.predict(vect)
    return render_template('result.html', prediction_text = my_prediction)


@app.route('/stemmer',methods=['POST', 'GET'])
def stemmer():
    if request.method == 'POST':
        message = request.form['message']
    else:
        message = request.args.get('message')
    text = stemm_text(message)
    return render_template('result.html',cleaned_text = """Stemmed Message: "{}" """.format(text), actual_text = "Message Entered: {}".format(message)) 

@app.route('/lemmatizer',methods=['POST', 'GET'])
def lemmatizer():
    if request.method == 'POST':
        message = request.form['message']
    else:
        message = request.args.get('message')
    text = lemmatize_text(message)
    return render_template('result.html',cleaned_text = """Lemmatized Message: "{}" """.format(text), actual_text = "Message Entered: {}".format(message)) 


if __name__ == '__main__':
	app.run(debug=True)
