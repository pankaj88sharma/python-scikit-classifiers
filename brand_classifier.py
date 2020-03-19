from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
import nltk
from nltk.stem import WordNetLemmatizer
from joblib import load
import re
from joblib import dump, load
nltk.download('wordnet')

def create_app():
    app = Flask(__name__)

    global model
    global cv
    global le

    try:
        model = load('model_brand.joblib')
        print('model loaded')

        le = load('le_brand.joblib')
        print('binarizer loaded')
        
        cv = load('cv_brand.joblib')
        print ('count vectorizer loaded')
    except FileNotFoundError:
        print('model files do not exists')

    CORS(app)
    return app

app = create_app()

@app.route('/brand_classifier/predict', methods=['GET'])
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
            model = load('model_brand.joblib')
            print('model reloaded')

            le = load('le_brand.joblib')
            print('binarizer reloaded')
        
            cv = load('cv_brand.joblib')
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


@app.route('/brand_classifier/train', methods=['GET'])
def train():

    df = pd.read_csv('http://localhost:8983/solr/clickstream/select?q=*:*&wt=csv&fl=keyword,brand&rows=9999999')
    df['TITLE'] = df['keyword']
    df['LABELS'] = df['brand']
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

    df['TITLE'] = df['TITLE'].map(lambda x : clean_text(x))
    df['TITLE'] = df['TITLE'].map(lambda x : stem_sentences(x))

    le_train = LabelEncoder()
    Y = le_train.fit_transform(df['LABELS'])
    cv_train = TfidfVectorizer()
    X = cv_train.fit_transform(df['TITLE'])

    nb_clf = MultinomialNB()
    clf = OneVsRestClassifier(nb_clf)
    clf.fit(X, Y)

    dump(clf, 'model_brand.joblib')
    dump(le_train, 'le_brand.joblib')
    dump(cv_train, 'cv_brand.joblib') 

    output = {}
    output['message'] = 'Model training completed. Files dumped succesfully.'

    return jsonify(output), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)