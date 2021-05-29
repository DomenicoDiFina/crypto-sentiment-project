import streamlit as st
#import pandas_datareader as web
import matplotlib.pyplot as plt
#import mplfinance as mpf
import datetime as dt

start = dt.datetime(2020,1,1)
end = dt.datetime.now()


st.title("Cryptocurrency Visualizer")

st.sidebar.write(""" ## Visualizzatore Criptovalute """)

crypto_name = st.sidebar.selectbox("Seleziona la crypto",('Bitcoin BTC', 'Ethereum ETH', 'Ripple XRP', 'Binance Coin BNB', 'Tether USDT', 'Cardano ADA', 'Dogecoin DOGE', 'Polkadot DOT', 'Internet Computer ICP', 'XRP', 'Uniswap UNI', 'Polygon MATIC', 'Stellar XLM', 'Litecoin LTC', 'VeChain VET', 'Solana SOL', 'SHIBA INU SHIB'))

if st.sidebar.button("Visualizza"):
    st.write(f"Grafico riguardante {crypto_name}.")

    # st.pyplot(fig) # per inserire il la figure matplotlib nella finestra