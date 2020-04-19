import flask
from flask import Flask, render_template, session, redirect, url_for, session
import numpy as np
# import joblib
import requests
import os
# import pickle as pkl
import praw
# import nltk
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# from sklearn.pipeline import make_pipeline
# from nltk.corpus import stopwords
import re
# import contractions
# import sklearn

app = flask.Flask(__name__, template_folder='templates', static_folder='assets')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('index.html')
    if flask.request.method == 'POST':
        url = flask.request.form['url']
        reddit = praw.Reddit(client_id='mZxKduOfxjNThA', \
                            client_secret='s9yiNq1s7URgalN4O8IHqOhCl9w', \
                            user_agent='rClassifier', \
                            username='RachitB2500', \
                            password='RacBan@1')
        
        sub = reddit.submission(url=url)
        data = [sub.title, sub.url, sub.selftext, sub.link_flair_text]
        # data[0] = process(data[0])
        # data[1] = processURL(data[1])
        # data[2] = process(data[2])

        # model = pkl.load(open('model_lr.pkl', 'rb'))
        # prediction = model.predict([data[0] + ' ' + data[2] + ' ' + data[1]])
        actual = data[3]

        return flask.render_template('index.html', pred = actual, act = actual)

# wnl = WordNetLemmatizer()
# remove =set(stopwords.words('english'))

# def process(words):
#     words = str(words)
#     words = re.sub('([.,////])',' ',words)
#     words = words.replace('\n', ' ')
#     try:
#       words = contractions.fix(words)
#     except:
#       pass
#     word_list = nltk.word_tokenize(re.sub(r'([^a-z A-Z])', '', words.lower()))
#     comment = ' '.join([wnl.lemmatize(w) for w in word_list if w not in remove])
    
#     return comment

# def processURL(words):
#     try:
#       words = words.split('://')[1]
#       words = words.split('/')
#       return ' '.join([' '.join(word.split('.')) for word in words])
#     except:
#       return words

# if __name__ == '__main__':
#     app.run(debug=True)