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
		st.title('About Climate Change')
		st.write('-----------------------------------------------')
		# st.info("General Information")

		# Intro
		
		st.markdown('## Introduction')
		if st.button("Click here to veiw Introduction"):
			st.info("""Climate change has been a trending topic ever since
				 Al Gore received a Nobel Peace Prize for his campaign in 2007.
				\n The topic has become a controversial subject on twitter where some 
				twitter users feel very strongly that climate change is not real 
				and is part of a conspiracy theory. To add fire to the situation, 
				American President, Donald Trump, claimed that climate change is a 
				Chinese-funded conspiracy. 
				\n As a result, some twitter users
				started tweeting that Climate Change is not real and trying to
				follow tweets about climate change suddenly required a degree in politics.""")

		# Problem Statement
		st.markdown('## Problem Statement')
		if st.button("Click here to veiw Problem Statement"):
			st.info("""Many companies are built around lessening one’s environmental 
					impact or carbon footprint. They offer products and services that 
					are environmentally friendly and sustainable, in line with their 
					values and ideals. They would like to determine how people perceive 
					climate change and whether or not they believe it is a real threat. 
					This would add to their market research efforts in gauging 
					how their product/service may be received""")

		st.subheader("The Climate Change Tweet Classifier aims to classify the sentiment of a tweet.")

		# Research
		st.markdown('## Research')
		if st.button("Click here to veiw Research"):
			st.info('Research Here')

		
		# EDA
		st.markdown('## Exploratory Data Analysis')
		if st.button("Click here to veiw EDA"):
			st.subheader('Exploratoring the following data')
			
			from PIL import Image
			image = Image.open('resources/imgs/i.PNG')

			st.image(image, caption=' Total classification of the four categories',
				use_column_width=True)

			st.subheader('Most tweeted hashtag')
			st.info("""\n anti : #MAGA (11) 
				\n neutral : #climate (16)
				\n pro : #climate : (130)
				\n news : #climate : (130)
				""")
			st.subheader('Most tweeted username')
			st.info("""\n anti : @realDonaldTrump (71)
				\n neutral : @StephenSchlegel (307)
				\n pro : @realDonaldTrump (31)
				\n news : @thehill (77)
				""")
			from PIL import Image
			image = Image.open('resources/imgs/a.PNG')

			st.image(image, caption=' Common Tweets',
				use_column_width=True)

			from PIL import Image
			image = Image.open('resources/imgs/b.PNG')

			st.image(image, caption=' Common Tweets',
				use_column_width=True)

			from PIL import Image
			image = Image.open('resources/imgs/c.PNG')

			st.image(image, caption=' ',
				use_column_width=True)

			from PIL import Image
			image = Image.open('resources/imgs/d.PNG')

			st.image(image, caption=' ',
				use_column_width=True)

			st.subheader('Infomation from data source')
			st.info("""\n Number of Original Tweets: 6133
				\n Number of Retweets: 9687)
				\n Ratio of Orignal Tweets to retweets: 0.63
				""")

			st.subheader('Exploring data Conclusion')
			st.info('We can see by this that there are alot more retweets in the data')
			
		# Adding word clouds		
		st.markdown('### Word Clouds')
		if st.button("Click here to veiw Word Clouds"):
			st.subheader("""These are the Word Clouds we created on the training set, the bigger the word the more common it occurs within the data""")
			st.info('Word Clouds Here')

		# Insights
		st.markdown('## Insights')
		if st.button("Click here to veiw Insights"):
			st.info('Insights Here')
		
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")
	    # add EDA
	    # add real world research

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write("show raw data") # will write the df to the page

	# Building out the predication page
	if selection == "Home":
		st.markdown("**This app will take input as text and return a classification into one of the four categories:**")
		st.info("""\n-1: Anti Climate Change
				\n0: Neutral towards Climate Change
				\n1: Pro Climate Change
				\n2: Factual News about Climate Change
				""")

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

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
    print('loading pickle file')
    model_load_path = "resources/linear_SVC.pkl"
    
    with open(model_load_path,'rb') as file:
        unpickled_model = pickle.load(file)

    print("model successfully pickled")
    main()