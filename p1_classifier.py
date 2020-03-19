from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import text
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
import re
from nltk.stem import WordNetLemmatizer
import nltk
from joblib import dump, load
nltk.download('wordnet')

def create_app():
    app = Flask(__name__)

    global model
    global cv
    global le

    try:
        model = load('model_p1.joblib')
        print('model loaded')

        le = load('le_p1.joblib')
        print('binarizer loaded')
        
        cv = load('cv_p1.joblib')
        print ('count vectorizer loaded')
    except FileNotFoundError:
        print('model files do not exists')

    CORS(app)
    return app

app = create_app()


@app.route('/p1_classifier/predict', methods=['GET'])
def predict():

    output = {}
    response = []
    lemmatizer = WordNetLemmatizer()

    global model
    global cv
    global le

    reload = request.args.get('reload')
    print(reload)

    if reload == 'true':
        model = None
        cv = None
        le = None
        try:
            model = load('model_p1.joblib')
            print('model reloaded')

            le = load('le_p1.joblib')
            print('binarizer reloaded')
        
            cv = load('cv_p1.joblib')
            print ('count vectorizer reloaded')
        except FileNotFoundError:
            print('model files do not exists')
            output['error'] = 'model file(s) are not available'
            return jsonify(output), 500


    if le is None or cv is None or model is None:
        output['error'] = 'model file(s) are not available'
        return jsonify(output), 500

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9_\s]+', '', text)
        text = text.strip(' ')
        return text
    
    def stem_sentences(sentence):
        tokens = sentence.split()
        stemmed_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(stemmed_tokens)

    cn = request.args.get('query')
    threshold = float(request.args.get('threshold'))

    query = stem_sentences(clean_text(cn))
    x_cn = cv.transform([query])

    result = [(a, b) for a, b in zip(le.classes_.tolist(), model.predict_proba(x_cn).tolist()[0]) if b > threshold]
        
    for x in result:
        res = {}
        res['value'] = x[0]
        res['score'] = round(x[1], 2)
        response.append(res)

    output['predictions'] = response

    return jsonify(output), 200


@app.route('/p1_classifier/train', methods=['GET'])
def train():

    df = pd.read_csv('processed_catalog.csv')
    df['LABELS'] = 'department:' + df['DEPARTMENT']
    col = ['TITLE', 'LABELS']
    df = df[col]

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9_\s]+', '', text)
        text = text.strip(' ')
        return text

    lemmatizer = WordNetLemmatizer()
    def stem_sentences(sentence):
        tokens = sentence.split()
        stemmed_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(stemmed_tokens)


    stop_words = text.ENGLISH_STOP_WORDS

    le_train = LabelEncoder()
    Y = le_train.fit_transform(df['LABELS'])
    cv_train = TfidfVectorizer(stop_words=stop_words)
    X = cv_train.fit_transform(df['TITLE'])

    nb_clf = MultinomialNB()
    clf = OneVsRestClassifier(nb_clf)
    clf.fit(X, Y)

    dump(clf, 'model_p1.joblib')
    dump(le_train, 'le_p1.joblib')
    dump(cv_train, 'cv_p1.joblib') 

    output = {}
    output['message'] = 'Model training completed. Files dumped succesfully.'

    return jsonify(output), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002)
