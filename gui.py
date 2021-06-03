import streamlit as st
#import pandas_datareader as web
import matplotlib.pyplot as plt
#import mplfinance as mpf
import datetime as dt
from datetime import date, timedelta
from get_tweets import get_topics, get_tweets, get_sentiment, get_emotion, create_plots, get_emotion_list, get_sentiment_list
from tqdm import tqdm
from streamlit import caching
import time

caching.clear_cache()

start = date.today() - timedelta(days=1)
end = date.today()




st.sidebar.write(""" ## Visualizzatore Criptovalute """)

crypto_name = st.sidebar.selectbox("Seleziona la crypto",('Bitcoin BTC', 'Ethereum ETH', 'Ripple XRP', 'Binance Coin BNB', 'Tether USDT', 'Cardano ADA', 'Dogecoin DOGE', 'Polkadot DOT', 'Internet Computer ICP', 'XRP', 'Uniswap UNI', 'Polygon MATIC', 'Stellar XLM', 'Litecoin LTC', 'VeChain VET', 'Solana SOL', 'SHIBA INU SHIB'))
date_start = st.sidebar.date_input('Data Iniziale', start)
date_end = st.sidebar.date_input('Data Finale', end)

limit_tweets = st.sidebar.select_slider('Seleziona il numero di tweet da recuperare per giorno', options=[100, 200, 300, 400])


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


if st.sidebar.button("Visualizza"):

    #emotion_list = list()
    #sentiment_list = list()
    st.title(f"Grafico riguardante {crypto_name} dal {date_start} al {date_end}.")
    print('Recupero tweet in corso...')
    progress_bar = st.progress(0.2)
    df = get_tweets('crypto', str(date_start), str(date_end), limit_tweets)
    print('Recupero tweet avvenuto')
    progress_bar.progress(0.5)
    progress_count = 0.5

    ''' NEW VERSION '''
    emotion_list = [(tweet['date'], get_emotion(tweet['processed_tweet'])) for _,tweet in df.iterrows() if crypto_name.lower() in get_topics(tweet['processed_tweet'])]
    sentiment_list = [(tweet['date'], get_sentiment(tweet['processed_tweet'])) for _,tweet in df.iterrows() if crypto_name.lower() in get_topics(tweet['processed_tweet'])]




    ''' OLD VERSION '''
    # emotion_list = list()
    # sentiment_list = list()
    # start = time.time()
    # for i in df.index:
    #     #print(f"i: {i}", df["processed_tweet"][i])
    #     #st.write(i)
    #     #st.write(df['processed_tweet'][i])

    #     progress_count += (50 / len(df.index)) / 100
    #     progress_bar.progress(round(progress_count,1))
    #     topics = get_topics(df['processed_tweet'][i])
    #     if topics != '':
    #         if crypto_name.lower() in topics:
    #             df['sentiment'][i] = get_sentiment(df['processed_tweet'][i])
    #             df['emotion'][i] = get_emotion(df['processed_tweet'][i])

    #             emotion_list.append((df["date"][i], df["emotion"][i]))
    #             sentiment_list.append((df["date"][i], df["sentiment"][i]))

    #             #st.write(i)
    #             #st.write(df["tweet"][i])
    #             #st.write(df['sentiment'][i])
    #             #st.write(df['emotion'][i])

    # end = time.time()
    # print('old_time: ',end - start)
    progress_bar.empty()

    emotion_figure, sentiment_figure, combined_figure = create_plots(emotion_list, sentiment_list)
    #st.write(df)

    st.pyplot(emotion_figure) 
    st.pyplot(sentiment_figure)
    st.pyplot(combined_figure)