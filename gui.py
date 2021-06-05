import streamlit as st
import matplotlib.pyplot as plt
import datetime as dt
from datetime import date, timedelta
from utils import get_topics, get_tweets, get_sentiment, get_emotion, create_plots
from tqdm import tqdm
import time

# remove default texts in the page
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

start = date.today() - timedelta(days=1)
end = date.today()

st.sidebar.write(""" ## Visualizzatore Criptovalute """)

crypto_name = st.sidebar.selectbox("Seleziona la crypto",('Bitcoin BTC', 'Ethereum ETH', 'Ripple XRP', 'Binance Coin BNB', 'Tether USDT', 'Cardano ADA', 'Dogecoin DOGE', 'Polkadot DOT', 'Internet Computer ICP', 'XRP', 'Uniswap UNI', 'Polygon MATIC', 'Stellar XLM', 'Litecoin LTC', 'VeChain VET', 'Solana SOL', 'SHIBA INU SHIB'))
date_start = st.sidebar.date_input('Data Iniziale', start)
date_end = st.sidebar.date_input('Data Finale', end)

# choose number of tweets to be retrieved per day
limit_tweets = st.sidebar.select_slider('Seleziona il numero di tweet da recuperare per giorno', options=[100, 200, 300, 400])


if st.sidebar.button("Visualizza"):

    st.title(f"Grafico riguardante {crypto_name} dal {date_start} al {date_end}.")
    
    progress_bar = st.progress(0.5)
    df = get_tweets('crypto', str(date_start), str(date_end), limit_tweets)
    
    progress_bar.progress(0.7)

    # list of (date, emotion) tuples
    emotion_list = [(tweet['date'], get_emotion(tweet['processed_tweet'])) for _,tweet in df.iterrows() if crypto_name.lower() in get_topics(tweet['processed_tweet'])]

    # list of (date, sentiment) tuples
    sentiment_list = [(tweet['date'], get_sentiment(tweet['processed_tweet'])) for _,tweet in df.iterrows() if crypto_name.lower() in get_topics(tweet['processed_tweet'])]

    progress_bar.empty()

    emotion_figure, sentiment_figure, combined_figure = create_plots(emotion_list, sentiment_list)

    st.pyplot(emotion_figure) 
    st.pyplot(sentiment_figure)
    st.pyplot(combined_figure)
