from json import load
from numpy.ma.core import masked_where
import twint
import pandas as pd
from datetime import datetime
from datetime import date, timedelta
import os
from pre_processing import pre_processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
import numpy as np
from keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

crypto = ['bitcoin BTC', 'ethereum ETH', 'Ripple XRP', 'Binance Coin BNB', 'Tether USDT', 'Cardano ADA', 'Dogecoin DOGE', 'Polkadot DOT', 'Internet Computer ICP', 'XRP', 'Uniswap UNI', 'Polygon MATIC', 'Stellar XLM', 'Litecoin LTC', 'VeChain VET', 'Solana SOL', 'SHIBA INU SHIB']
crypto = [c.lower() for c in crypto]
tf_idf_vectorizer = TfidfVectorizer()
crypto_vectorized = tf_idf_vectorizer.fit_transform(crypto)

model_sentiment = load_model('model_lstm_epoch_1.h5')
tokenizer_sentiment = pickle.load(open("tokenizer.pickle", "rb"))

model_emotion = load_model("emotions_model_lstm")
tokenizer_emotion = pickle.load(open("emotion_detection/tokenizer_emotion.pickle", "rb"))


def get_tweets(topic, start_date, end_date, limit):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)


    date_range = list(daterange(start_date, end_date))

    for index in range(len(date_range)-1):
        c = twint.Config()
        c.Search = topic
        c.Limit = limit
        c.Since = date_range[index].strftime("%Y-%m-%d")
        c.Until = date_range[index+1].strftime("%Y-%m-%d")
        c.Verified = True # True se vogliamo i verificati, False altrimenti
        c.Output = f"./tweets.csv"
        c.Store_csv = True
        c.Hide_output = True
        twint.run.Search(c)
    

    df = pd.read_csv('tweets.csv')

    df = df[df['language'] == 'en']

    df['processed_tweet'] = pre_processing(list(df['tweet']))

    df["processed_tweet"] = df['processed_tweet'].apply(lambda x: ' '.join(map(str,x)))

    df["sentiment"] = ''
    
    df["emotion"] = ''

    os.remove('tweets.csv')

    return df.loc[:,['date', 'time', 'tweet', 'language', 'processed_tweet', 'sentiment', 'emotion']]
    


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)
    


def get_sentiment(tweet):
    
    pred = model_sentiment.predict(pad_sequences(tokenizer_sentiment.texts_to_sequences(tweet), maxlen=48, dtype='int32', value=0))[0]
    if(np.argmax(pred) == 0):
        return -1
    elif (np.argmax(pred) == 1):
        return 1


"""
UPDATE WITH NEW RAPPRESENTATION BASED ON NEW EMOTIONS MODEL
"""
def get_emotion(tweet):
    pred = model_emotion.predict(pad_sequences(tokenizer_emotion.texts_to_sequences(tweet), maxlen=30, dtype='int32', value=0))[0]
    if(np.argmax(pred) == 0):
        return "happiness"
    elif(np.argmax(pred) == 1):
        return "love"
    elif(np.argmax(pred) == 2):
        return "neutral"
    elif(np.argmax(pred) == 3):
        return "sad"
    elif(np.argmax(pred) == 4):
        return "worry"

def get_topics(tweet):
    tweet_vectorized = tf_idf_vectorizer.transform([tweet])    
    cosine_sim = cosine_similarity(crypto_vectorized, tweet_vectorized)

    if np.max(cosine_sim) > 0:
        return [crypto[i] for i, value in enumerate(cosine_sim) if value > 0]
    else:
        return ''



def create_emotion_plot(emotion_list):

    date_list = sorted(list(set([datetime.strptime(day[0], '%Y-%m-%d') for day in emotion_list])))
    start_date = date_list[0]
    end_date = date_list[-1] + timedelta(days=1)

    x = [x.strftime("%Y-%m-%d") for x in list(daterange(start_date, end_date))]
    """
    0: happiness
    1: love
    2: neutral
    3: sad
    4: worry
    """
    y = {}
    y["happiness"] = np.zeros(len(x))
    y["love"] = np.zeros(len(x))
    y["neutral"] = np.zeros(len(x))
    y["sad"] = np.zeros(len(x))
    y["worry"] = np.zeros(len(x))

    for day in range(0, len(x)):
        for emotion in emotion_list:
            if emotion[0] == x[day]:
                y[emotion[1]][day] += 1

    plt.style.use('dark_background')
    
    fig, ax = plt.subplots()
    fig.patch.set_alpha(0)

    fig.suptitle(f'Emotions Counter dal {start_date.strftime("%d-%m-%Y")} al {end_date.strftime("%d-%m-%Y")}')
    ax.bar(x, y["happiness"], label='happy', color='#003f5c')
    ax.bar(x, y["love"], bottom=y["happiness"], label='love', color='#58508d')
    ax.bar(x, y["neutral"], bottom=y["love"]+y["happiness"],label='neutral', color='#bc5090')
    ax.bar(x, y["sad"], bottom=y["love"]+y["happiness"]+y["neutral"], label='sad', color='#ff6361')
    ax.bar(x, y["worry"],bottom=y["love"]+y["happiness"]+y["neutral"]+y["sad"], label='worry', color='#ffa600')
    ax.legend()
    return fig


def create_sentiment_plot(sentiment_list):

    date_list = sorted(list(set([datetime.strptime(day[0], '%Y-%m-%d') for day in sentiment_list])))
    start_date = date_list[0]
    end_date = date_list[-1] + timedelta(days=1)

    x = [x.strftime("%Y-%m-%d") for x in list(daterange(start_date, end_date))]
    y = list()
    for day in x:
        y.append(0)
        for sentiment in sentiment_list:
            if sentiment[0] == day:
                y[-1] += sentiment[1]

    #plotting   
    plt.style.use('dark_background')
    fig = plt.figure()
    fig.patch.set_alpha(0)
    ax = fig.add_subplot(1, 1, 1)
    #ax.style.use('fivethirtyeight')
    ax.plot(x, y)

    fig.suptitle(f'Sentiment dal {start_date.strftime("%d-%m-%Y")} al {end_date.strftime("%d-%m-%Y")}')

    return fig


def create_plots(emotion_list, sentiment_list):

    return create_sentiment_plot(sentiment_list), create_emotion_plot(emotion_list), create_sentiment_plot(sentiment_list)