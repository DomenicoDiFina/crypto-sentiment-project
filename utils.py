import twint
import pandas as pd
from datetime import datetime
from datetime import timedelta
import os
from pre_processing import pre_processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from keras.models import load_model
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# list of crypto
crypto = ['bitcoin BTC', 'ethereum ETH', 'Ripple XRP', 'Binance Coin BNB', 'Tether USDT', 'Cardano ADA', 'Dogecoin DOGE', 'Polkadot DOT', 'Internet Computer ICP', 'XRP', 'Uniswap UNI', 'Polygon MATIC', 'Stellar XLM', 'Litecoin LTC', 'VeChain VET', 'Solana SOL', 'SHIBA INU SHIB']
crypto = [c.lower() for c in crypto]

# tf-idf vectorizer used to create a numeric vector for every crypto
tf_idf_vectorizer = TfidfVectorizer()
crypto_vectorized = tf_idf_vectorizer.fit_transform(crypto)

# load the sentiment model created using a neural network with a lstm layer
model_sentiment = load_model('model_lstm_epoch_1.h5')

# load the tokenizer for the sentiments
tokenizer_sentiment = pickle.load(open("tokenizer_sentiment.pickle", "rb"))

# load the emotion model created using a neural network with a lstm layer
model_emotion = load_model("emotions_model_lstm")

# load the tokenizer for the emotions
tokenizer_emotion = pickle.load(open("tokenizer_emotion.pickle", "rb"))

# weights for every emotion
emotions_dict = {
    'neutral' : 1,
    'happiness' : 2,
    'love' : 3,
    'worry' : 2,
    'sad' : 3
}


# retrieve {limit} tweets per day from {start_date} to {end_date} about a particular {topic}
def get_tweets(topic, start_date, end_date, limit):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)

    date_range = list(daterange(start_date, end_date))

    # twint library used to retrieve tweets per day (we consider only verified accounts)
    for index in range(len(date_range)):
        c = twint.Config()
        c.Search = topic
        c.Limit = limit
        c.Since = date_range[index].strftime("%Y-%m-%d") + " 00:00:00"
        c.Until = date_range[index].strftime("%Y-%m-%d") + " 21:59:59"
        c.Verified = True 
        c.Output = f"./tweets.csv"
        c.Store_csv = True
        c.Hide_output = True
        twint.run.Search(c)
    

    df = pd.read_csv('tweets.csv')

    # check if the language of the tweet is english
    df = df[df['language'] == 'en']

    # preprocessing
    df['processed_tweet'] = pre_processing(list(df['tweet']))

    # convert into a string every row tokenized tweet
    df["processed_tweet"] = df['processed_tweet'].apply(lambda x: ' '.join(map(str,x)))

    df["sentiment"] = ''
    
    df["emotion"] = ''

    # delete the csv file
    os.remove('tweets.csv')

    return df.loc[:,['date', 'time', 'tweet', 'language', 'processed_tweet', 'sentiment', 'emotion']]
    

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)
    

# using the neural network model to predict the sentiment of a tweet
def get_sentiment(tweet): 
    pred = model_sentiment.predict(pad_sequences(tokenizer_sentiment.texts_to_sequences(tweet), maxlen=48, dtype='int32', value=0))[0]
    if(np.argmax(pred) == 0):
        return -1
    elif (np.argmax(pred) == 1):
        return 1

# using the neural network model to predict the predominant emotion in a tweet
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


# return the list of topics in a tweet
def get_topics(tweet):
    tweet_vectorized = tf_idf_vectorizer.transform([tweet])    
    cosine_sim = cosine_similarity(crypto_vectorized, tweet_vectorized)

    if np.max(cosine_sim) > 0:
        return [crypto[i] for i, value in enumerate(cosine_sim) if value > 0]
    else:
        return ''


def create_emotion_plot(emotion_list):

    date_list = sorted(list(set([datetime.strptime(day[0], '%Y-%m-%d') for day in emotion_list])))
    print('date_list: ', date_list)
    start_date = date_list[0]
    end_date = date_list[-1] + timedelta(days=1)

    x = [x.strftime("%Y-%m-%d") for x in list(daterange(start_date, end_date))]
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

    x = [x.strftime("%d-%m") for x in list(daterange(start_date, end_date))]
    fig.suptitle(f'Emotions Counter dal {start_date.strftime("%d-%m-%Y")} al {(end_date - timedelta(days=1)).strftime("%d-%m-%Y")}')
    ax.bar(x, y["happiness"], label='happy', color='#003f5c')
    ax.bar(x, y["love"], bottom=y["happiness"], label='love', color='#58508d')
    ax.bar(x, y["neutral"], bottom=y["love"]+y["happiness"],label='neutral', color='#bc5090')
    ax.bar(x, y["sad"], bottom=y["love"]+y["happiness"]+y["neutral"], label='sad', color='#ff6361')
    ax.bar(x, y["worry"],bottom=y["love"]+y["happiness"]+y["neutral"]+y["sad"], label='worry', color='#ffa600')
    ax.legend()
    
    plt.xticks(rotation=90)
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

    x = [x.strftime("%d-%m") for x in list(daterange(start_date, end_date))]
   
    plt.style.use('dark_background')
    fig = plt.figure()
    fig.patch.set_alpha(0)
    ax = fig.add_subplot(1, 1, 1)

    
    ax.plot(x, y)
    yabs_max = abs(max(ax.get_ylim(), key=abs))
    ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    plt.xticks(rotation=90)

    fig.suptitle(f'Sentiment dal {start_date.strftime("%d-%m-%Y")} al {(end_date - timedelta(days=1)).strftime("%d-%m-%Y")}')

    return fig


def create_combined_plot(sentiment_list, emotion_list):

    date_list = sorted(list(set([datetime.strptime(day[0], '%Y-%m-%d') for day in sentiment_list])))
    start_date = date_list[0]
    end_date = date_list[-1] + timedelta(days=1)

    x = [x.strftime("%Y-%m-%d") for x in list(daterange(start_date, end_date))]
    y = list()
    for day in x:
        y.append(0)
        for index, sentiment in enumerate(sentiment_list):
            if sentiment[0] == day:
                y[-1] += sentiment[1] * emotions_dict[emotion_list[index][1]]


    x = [x.strftime("%d-%m") for x in list(daterange(start_date, end_date))]
   
    plt.style.use('dark_background')
    fig = plt.figure()
    fig.patch.set_alpha(0)
    ax = fig.add_subplot(1, 1, 1)
    
    ax.plot(x, y)
    yabs_max = abs(max(ax.get_ylim(), key=abs))
    ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    plt.xticks(rotation=90)


    fig.suptitle(f'Sentiment e Emozioni Combinate dal {start_date.strftime("%d-%m-%Y")} al {(end_date - timedelta(days=1)).strftime("%d-%m-%Y")}')

    return fig


def create_plots(emotion_list, sentiment_list):

    return create_sentiment_plot(sentiment_list), create_emotion_plot(emotion_list), create_combined_plot(sentiment_list, emotion_list)
