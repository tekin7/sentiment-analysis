
#kütüphanelerin eklenmesi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Dosyaları okuma

train = pd.read_csv("train.tsv", sep="\t")
test = pd.read_csv("test.tsv", sep="\t")

#Veri setini gereksiz kelimelerden kurtardık

stop_word = set(stopwords.words('english')) 
word_vectorizer = CountVectorizer(ngram_range=(1,1), analyzer='word', min_df=0.001)

#Tekrar eden kelimeleri bir kere kullanmak için sparse matrix kullandık

sparse_matrix = word_vectorizer.fit_transform(test['Phrase'])
frequencies = sum(sparse_matrix).toarray()[0]
freq = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency'])
freq.sort_values('frequency', ascending=False)

#veri setini gereksiz karakterlerden arındırdık ve bütün harfleri küçük harf yaptık

train['Phrase'] = train['Phrase'].str.lower()
train['Phrase'] = train['Phrase'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
test['Phrase'] = test['Phrase'].str.lower()
test['Phrase'] = test['Phrase'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

#Veri setindeki cümleleri kelimelere ayırdık

X_train = train.Phrase
y_train = test.Sentiment
tokenize = Tokenizer()
tokenize.fit_on_texts(X_train.values)

#verisetinde ki kelimeleri sayısal olarak temsil ettik

X_test = test.Phrase
X_train = tokenize.texts_to_sequences(X_train)
X_test = tokenize.texts_to_sequences(X_test)

#Verileri aynı uzunluğa getirdik

max_lenght = max([len(s.split()) for s in train['Phrase']])
X_train = pad_sequences(X_train, max_lenght)
X_test = pad_sequences(X_test, max_lenght)


#neural network mimarisini oluşturduk

EMBEDDING_DIM = 100
unknown = len(tokenize.word_index)+1
model = Sequential()
model.add(Embedding(unknown, EMBEDDING_DIM, input_length=max_lenght))
model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2 ))
model.add(Dense(5, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#modeli fit ettik

model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=1)

#tahmin

prediction = model.predict_classes(X_test)
