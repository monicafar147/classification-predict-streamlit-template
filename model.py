# general
import numpy as np 
import pandas as pd
import pickle
import joblib,os

# text preprocessing
import re
from string import punctuation
import nltk
nltk.download(['stopwords','punkt'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# models
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.svm import LinearSVC

# metrics
from sklearn.metrics import classification_report

def preprocess(tweet):
  tweet = tweet.lower() # converting everything to lower case
  tweet = re.sub(r"\W", " ", tweet)
  tweet = re.sub(r'#([^\s]+)', r'\1', tweet) 
  tweet = word_tokenize(tweet) # tokenizing the tweets so that they may be used in modeling and/or NLP
  stopwords_list = set(stopwords.words('english') + list(punctuation)) # cresting a list containing punctuation and stop words
  tweets = [word for word in tweet if word not in stopwords_list] # iterating over the list and saving the output into a list 
  return " ".join(tweet)

if __name__ == '__main__':

    # load in data
    print('loading in data')
    train = pd.read_csv('https://raw.githubusercontent.com/monicafar147/classification-predict-streamlit-template/master/climate-change-belief-analysis/train.csv')
    print(train.head())
    train['processed'] = train['message'].apply(preprocess)
    print('data preprocessed')
    #print(train.head())

    # train test split
    X = train['processed']
    y = train['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state =10)

    # apply model on train data
<<<<<<< HEAD
    #creating a pipeline with the tfid vectorizer and a linear svc model
=======
    # creating a pipeline with the tfid vectorizer and a linear svc model
>>>>>>> streamlit-features
    svc = Pipeline([('tfidf',TfidfVectorizer()),('classify',LinearSVC())])

    #fitting the model
    svc.fit(X_train, y_train)

    #apply model on test data
    y_pred_svc = svc.predict(X_test)

    # Print a classification report
    print('report on y_test vs svc.predict(X_test)')
    print(classification_report(y_test, y_pred_svc))

    # pickle model
    model_save_path = "resources/linear_SVC.pkl"
    with open(model_save_path,'wb') as file:
        pickle.dump(svc,file)

    print('loading pickle file')
    model_load_path = "resources/linear_SVC.pkl"
    
    with open(model_load_path,'rb') as file:
        unpickled_model = pickle.load(file)

<<<<<<< HEAD
    # load test data in
    test = pd.read_csv('https://raw.githubusercontent.com/monicafar147/classification-predict-streamlit-template/master/climate-change-belief-analysis/test.csv')
    test['processed'] = test['message'].apply(preprocess)
    X_unseen = test['processed']

    y_unseen = unpickled_model.predict(X_unseen)
    print('model loaded: True')
    print('y_unseen')
    print(y_unseen)

=======
>>>>>>> streamlit-features
    tweet = "china is to blame for climate change! #die #flood"
    new = preprocess(tweet)
    print(new)
    tweet_pred = unpickled_model.predict([new])
<<<<<<< HEAD
    print("predicted",tweet_pred)

=======
    print("predicted",tweet_pred)
>>>>>>> streamlit-features
