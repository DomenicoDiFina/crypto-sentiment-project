# Crypto Sentiment Project

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://github.com/DomenicoDiFina/crypto-sentiment-project)

# Introduzione

Crypto sentiment project nasce con l'idea di applicare tecniche di text processing, sentiment detection ed emotion detection. Per far ciò a scopo puramente didattico è stato creato questo progetto, che consiste nell'acquisizione di tweets in tempo reale per estrarne emozioni e sentimenti per prevedere l'andamento di una particolare cripto valuta selezionata dall'utente.

Il progetto è diviso in quattro fasi:
- Acquisizione dati
- pre-processing dati
- Creazione modelli di machine learning ed information retreival
- Sviluppo interfaccia web per l'utente


# Acquisizione dati

Dovendo addestrare due differenti modelli di machine learning per effettuare il riconoscimento di emozioni e sentimenti sono stati utilizzati i seguenti dataset disponibili open-source:

- https://www.kaggle.com/kazanova/sentiment140
- https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp
- https://www.kaggle.com/pashupatigupta/emotion-detection-from-text

# Pre processing dati

Questa fase comprende tutta l'analisi dei dataset descritti precedentemente e l'opportuna fase di processing effettuata su tutti i tweet ricavati dai dataset. Nello specifico dai dataset sono state eliminate tutte quelle colonne ritenute inutili ai fini statistici e di addestramento dei modelli, quali username della persona che scriveva il tweet, orario, etc. Mentre per tutta la fase di processing dei tweet è possibile vedere il file [pre-processing] per maggiori informazioni

# Creazione modelli di machine learning ed information retreival

I modelli creati sono tre. Il primo modello di machine learning per il sentiment detection creato mediante rete LSTM avente la seguente struttura:

```python
embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(30000, 300,input_length = X.shape[1]))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='rmsprop',metrics = ['accuracy'])
print(model.summary())
```
Con il quale si è ottenuta una accuracy sul validation test pari a: 78%
Per maggiori dettagli vedere il notebook: [sentiment_notebook]

Il secondo modello sempre basato su una rete LSTM invece possiede la seguente struttura:

```python
embed_dim = 300
lstm_out = 128

model = Sequential()
model.add(Embedding(5000, 300, input_length = X.shape[1]))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(5,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop',metrics = ['accuracy'])
print(model.summary())
```

Con il quale si è ottenuta una accuracy sul validation test pari a: 82%
Per maggiori dettagli vedere il notebook: [emotion_model]

Mentre il terzo modello è stato sviluppato per effettuare information retreival sui tweet forniti in input per riconoscere il topic di interesse, nello specifico riconoscere se all'interno di un dato tweet si parlava di una certa cripto valuta, utilizzando come funzione di tokenizzazione TF-IDF.

```python
crypto = ['bitcoin BTC', 'ethereum ETH', 'Ripple XRP', 'Binance Coin BNB', 'Tether USDT', 'Cardano ADA', 'Dogecoin DOGE', 'Polkadot DOT', 'Internet Computer ICP', 'XRP', 'Uniswap UNI', 'Polygon MATIC', 'Stellar XLM', 'Litecoin LTC', 'VeChain VET', 'Solana SOL', 'SHIBA INU SHIB']
crypto = [c.lower() for c in crypto]

# tf-idf vectorizer used to create a numeric vector for every crypto
tf_idf_vectorizer = TfidfVectorizer()
crypto_vectorized = tf_idf_vectorizer.fit_transform(crypto)
```
Per maggiori dettagli vedere il file: [utils]

# Sviluppo interfaccia web per l'utente

Per creare l'interfaccia web si è utilizzata la libreria [streamlit] che ha permesso in modo rapido e veloce di costruire una interfaccia responsive ed immediata. Tutti i dettagli in merito all'implementazione si trovano all'interno del file [gui]


**Free Software**

   [pre-processing]:  <https://github.com/DomenicoDiFina/crypto-sentiment-project/blob/main/pre_processing.py>
   [sentiment_notebook]: <https://github.com/DomenicoDiFina/crypto-sentiment-project/blob/main/sentiment_analysis_model.ipynb>
   [emotion_model]: <https://github.com/DomenicoDiFina/crypto-sentiment-project/blob/main/emotion_detection/emotion_model.ipynb>
   [utils]: <https://github.com/DomenicoDiFina/crypto-sentiment-project/blob/main/utils.py>
   [streamlit]: <https://streamlit.io/>
   [gui]: <https://github.com/DomenicoDiFina/crypto-sentiment-project/blob/main/gui.py>
