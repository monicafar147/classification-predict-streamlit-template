# Helper Dependencies
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None 
import pickle


# cleaning text
import re
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords 
#nltk.download('stopwords')
#nltk.download('wordnet')
from sklearn.model_selection import train_test_split

# Machine learning models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# metrics
from sklearn.metrics import confusion_matrix,classification_report


def _preprocess(tweet):
    stopwords_list = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])
    tweet = tweet.lower() # convert text to lower-case
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
    tweet = re.sub(r"\W", " ", tweet) # remove usernames
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
    tweet = word_tokenize(tweet) # remove repeated characters (helloooooooo into hello)
    tweets = [word for word in tweet if word not in stopwords_list]
    return " ".join(tweets) 

if __name__ == '__main__':

    # load in data
    print('loading in data')
    train = pd.read_csv('https://raw.githubusercontent.com/monicafar147/classification-predict-streamlit-template/master/climate-change-belief-analysis/train.csv')
    print(train.head())
    train['processed'] = train['message'].apply(_preprocess)
    print('data preprocessed')
    #print(train.head())

    # train test split
    X = train['processed']
    y = train['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state =10)

    # apply model on train data
    # Linear SVC:
    svc = Pipeline([('tfidf',TfidfVectorizer()),('classify',LinearSVC())])
    svc.fit(X_train, y_train)

    #apply model on test data
    pred = svc.predict(X_test)

    # Print a classification report
    print('report on y_test vs svc.predict(X_test)')
    print(classification_report(y_test, pred))

    # pickle model
    #model_save_path = "r"esources/svc.pkl"svc.pkl"
    #with open(model_save_path,'wb') as file:
    #    pickle.dump(svc,file)

    print('loading pickle file')
    model_load_path = "resources/svc.pkl"
    
    with open(model_load_path,'rb') as file:
        unpickled_model = pickle.load(file)

    # load test data in
    test = pd.read_csv('https://raw.githubusercontent.com/monicafar147/classification-predict-streamlit-template/master/climate-change-belief-analysis/train.csv')
    test['processed'] = test['message'].apply(_preprocess)
    X_unseen = test['processed']

    print('model loaded: True')
    y_unseen = unpickled_model.predict(X_unseen)
    print('y_unseen')
    print(y_unseen)
