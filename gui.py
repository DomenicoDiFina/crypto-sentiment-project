import streamlit as st
#import pandas_datareader as web
import matplotlib.pyplot as plt
#import mplfinance as mpf
import datetime as dt
from datetime import date, timedelta
from get_tweets import get_topics, get_tweets, get_sentiment, get_emotion, create_plots

start = date.today() - timedelta(days=1)
end = date.today()


st.title("Cryptocurrency Visualizer")

st.sidebar.write(""" ## Visualizzatore Criptovalute """)

crypto_name = st.sidebar.selectbox("Seleziona la crypto",('Bitcoin BTC', 'Ethereum ETH', 'Ripple XRP', 'Binance Coin BNB', 'Tether USDT', 'Cardano ADA', 'Dogecoin DOGE', 'Polkadot DOT', 'Internet Computer ICP', 'XRP', 'Uniswap UNI', 'Polygon MATIC', 'Stellar XLM', 'Litecoin LTC', 'VeChain VET', 'Solana SOL', 'SHIBA INU SHIB'))
date_start = st.sidebar.date_input('Data Iniziale', start)
date_end = st.sidebar.date_input('Data Finale', end)


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


if st.sidebar.button("Visualizza"):
    emotion_list = list()
    sentiment_list = list()
    st.write(f"Grafico riguardante {crypto_name} dal {date_start} al {date_end}.")
    df = get_tweets('crypto', str(date_start), str(date_end), 100)
    for i in df.index:
        #print(f"i: {i}", df["processed_tweet"][i])
        #st.write(i)
        #st.write(df['processed_tweet'][i])
        topics = get_topics(df['processed_tweet'][i])
        if topics != '':
            if crypto_name.lower() in topics:
                #print(df['processed_tweet'][i])
                #st.write(get_sentiment(df['processed_tweet'][i]))
                df['sentiment'][i] = get_sentiment(df['processed_tweet'][i])
                df['emotion'][i] = get_emotion(df['processed_tweet'][i])

                emotion_list.append((df["date"][i], df["emotion"][i]))
                sentiment_list.append((df["date"][i], df["sentiment"][i]))

                #st.write(i)
                #st.write(df["tweet"][i])
                #st.write(df['sentiment'][i])
                #st.write(df['emotion'][i])

    emotion_figure, sentiment_figure, combined_figure = create_plots(emotion_list, sentiment_list)
    #st.write(df)

    st.pyplot(emotion_figure) 
    st.pyplot(sentiment_figure)
    st.pyplot(combined_figure)