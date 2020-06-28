"""
    Simple Streamlit webserver application for serving developed classification
	models.
    Author: Explore Data Science Academy.
    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------
    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.
	For further help with the Streamlit framework, see:
	https://docs.streamlit.io/en/latest/
"""

# Streamlit dependencies
import streamlit as st
import joblib,os

# general
import numpy as np 
import pandas as pd
import pickle

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

# function to preprocess tweets
def preprocess(tweet):
  tweet = tweet.lower() # converting everything to lower case
  tweet = re.sub(r"\W", " ", tweet)
  tweet = re.sub(r'#([^\s]+)', r'\1', tweet) 
  tweet = word_tokenize(tweet) # tokenizing the tweets so that they may be used in modeling and/or NLP
  stopwords_list = set(stopwords.words('english') + list(punctuation)) # cresting a list containing punctuation and stop words
  tweets = [word for word in tweet if word not in stopwords_list] # iterating over the list and saving the output into a list 
  return " ".join(tweet)

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Classifing tweets towards their belief in Climate Change")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home", "About","Modelling"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "About":
		# Title
		st.title('About')
		st.write('-----------------------------------------------')
		# st.info("General Information")

		# Intro
		st.markdown('## Introduction')
		st.info('Intro Here')

		# Research
		st.markdown('## Research')
		st.info('Research Here')

		# EDA
		st.markdown('## Exploratory Data Analysis')

			# Adding word clouds
		st.info("These are the Word Clouds we created on the training set, the bigger the word the more common it occurs within the data")
		from PIL import Image
		pro = Image.open('resources/imgs/wordcloud_anti.PNG')
		st.image(pro)
		# Insights
		st.markdown('## Insights')
		st.info('Insights Here')
		
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")
	    # add real world research

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write("show raw data") # will write the df to the page

	# Building out the predication page
	if selection == "Home":
		st.markdown("**This app will take input as text and return a classification into one of the four categories:**")
		st.write("(see the About page for more information)")

		# Creating a text box for user input
		st.markdown("### Enter Text Bellow")
		tweet_text = st.text_area("","Type Here")

		if st.button("Classify Linear SVC model"):
			tweet = tweet_text
			print("input tweet : \n {}".format(tweet))
			tweet_processed = preprocess(tweet)
			print("processed tweet : \n {}".format(tweet_processed))
			tweet_pred = unpickled_model.predict([tweet_processed])
			print("predicted",tweet_pred)
			st.success("Text Categorized as: {}".format(tweet_pred))

	# add choose a model button
		if st.button("Classify LR model"):
			tweet = tweet_text
			print("input tweet : \n {}".format(tweet))
			tweet_processed = preprocess(tweet)
			print("processed tweet : \n {}".format(tweet_processed))
			tweet_pred = unpickled_model.predict([tweet_processed])
			print("predicted",tweet_pred)
			st.success("LR Model")

	if selection == "Modelling":
		# title
		st.title('Modelling process')
		st.write("""for the modeling process the outputs that we were 
				trying to predict were based on the following keys for a sentiment column:""")
		st.info("""\n-1: Anti Climate Change
				\n0: Neutral towards Climate Change
				\n1: Pro Climate Change
				\n2: Factual News about Climate Change
				""")

		# preprocessing
		st.markdown("## Preprocessing the Data")
		st.info('preprocessing here')

		# Model 1
		st.markdown("## Model 1")
		st.info('Model 1 info here')

		# Model 2
		st.markdown('## Model 2')
		st.info('Model 2 info here')

		# Model 3
		st.markdown('## Model 3')
		st.info(" Model 3 info here")

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
    print('loading pickle file')
    model_load_path = "resources/linear_SVC.pkl"
    
    with open(model_load_path,'rb') as file:
        unpickled_model = pickle.load(file)

    print("model successfully pickled")
    main()