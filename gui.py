import streamlit as st
#import pandas_datareader as web
import matplotlib.pyplot as plt
#import mplfinance as mpf
import datetime as dt
from datetime import date, timedelta

start = date.today() - timedelta(days=1)
end = date.today()


st.title("Cryptocurrency Visualizer")

st.sidebar.write(""" ## Visualizzatore Criptovalute """)

crypto_name = st.sidebar.selectbox("Seleziona la crypto",('Bitcoin BTC', 'Ethereum ETH', 'Ripple XRP', 'Binance Coin BNB', 'Tether USDT', 'Cardano ADA', 'Dogecoin DOGE', 'Polkadot DOT', 'Internet Computer ICP', 'XRP', 'Uniswap UNI', 'Polygon MATIC', 'Stellar XLM', 'Litecoin LTC', 'VeChain VET', 'Solana SOL', 'SHIBA INU SHIB'))
date_start = st.sidebar.date_input('Data Iniziale', start)
date_end = st.sidebar.date_input('Data Finale', end)

if st.sidebar.button("Visualizza"):
    st.write(f"Grafico riguardante {crypto_name} dal {date_start} al {date_end}.")

    # st.pyplot(fig) # per inserire il la figure matplotlib nella finestra