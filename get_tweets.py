from json import load
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

crypto = ['bitcoin BTC', 'ethereum ETH', 'Ripple XRP', 'Binance Coin BNB', 'Tether USDT', 'Cardano ADA', 'Dogecoin DOGE', 'Polkadot DOT', 'Internet Computer ICP', 'XRP', 'Uniswap UNI', 'Polygon MATIC', 'Stellar XLM', 'Litecoin LTC', 'VeChain VET', 'Solana SOL', 'SHIBA INU SHIB']
crypto = [c.lower() for c in crypto]
tf_idf_vectorizer = TfidfVectorizer()
crypto_vectorized = tf_idf_vectorizer.fit_transform(crypto)

model_sentiment = load_model('model_lstm_epoch_1.hd5')
tokenizer_sentiment = pickle.load(open("tokenizer.pickle", "rb"))


print(model_sentiment.summary())

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
    print("len: ", len(tokenizer_sentiment.texts_to_sequences(tweet)))
    x = np.asarray(tokenizer_sentiment.texts_to_sequences(tweet)).astype('float32')
    pred = model_sentiment.predict(x)[0]
    print("prediction: ", pred)
    if(np.argmax(pred) == 0):
        return "negative"
    elif (np.argmax(pred) == 1):
        return "positive"

def get_topics(tweet):
    tweet_vectorized = tf_idf_vectorizer.transform([tweet])    
    cosine_sim = cosine_similarity(crypto_vectorized, tweet_vectorized)

    if np.max(cosine_sim) > 0:
        return [crypto[i] for i, value in enumerate(cosine_sim) if value > 0]
    else:
        return ''