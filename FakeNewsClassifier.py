import nltk

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
# import keras
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
from tensorflow.keras.models import Model

true_df=pd.read_csv("/Users/owen/Downloads/True.csv")
fake_df=pd.read_csv("/Users/owen/Downloads/Fake.csv")

true_df.info()
fake_df.info()
true_df.isnull().sum()
fake_df.isnull().sum()

true_df['isfake'] = 0
fake_df['isfake'] = 1
df = pd.concat([true_df, fake_df]).reset_index(drop = True)
df.drop(columns = ['date'], inplace = True)
df['original'] = df['title'] + ' ' + df['text']

#Data Cleaning
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
stopwords.extend(['from', 'subject', 're', 'edu', 'use'])


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stopwords:
            result.append(token)
    return result

df['clean'] = df['original'].apply(preprocess)
df.head()

words_list = []
for i in df.clean:
    for j in i:
        words_list.append(j)

len(words_list)
total_words = len(list(set(words_list)))
total_words

df['clean_joined'] = df['clean'].apply(lambda x: " ".join(x))

#Visualize Cleaned Data

plt.figure(figsize = (8,8))
sns.countplot(y="subject",data=df)

plt.figure(figsize = (8,8))
sns.countplot(y="isfake",data=df)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.clean_joined, df.isfake, test_size = 0.2)

from nltk import word_tokenize
tokenizer = Tokenizer(num_words = total_words)
tokenizer.fit_on_texts(x_train)
train_seq= tokenizer.texts_to_sequences(x_train)
test_seq = tokenizer.texts_to_sequences(x_test)
print("The encoding for document\n",df.clean_joined[0],"\n is : ",train_seq[0])

pad_train = pad_sequences(train_seq,maxlen = 40, padding = 'post', truncating = 'post')
pad_test = pad_sequences(test_seq,maxlen = 40, truncating = 'post')

for i,doc in enumerate(pad_train[:2]):
     print("The padded encoding for document",i+1," is : ",doc)


model = Sequential()
model.add(Embedding(total_words, output_dim = 128))
model.add(Bidirectional(LSTM(128)))

model.add(Dense(128, activation = 'relu'))
model.add(Dense(1,activation= 'sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()

y_train = np.asarray(y_train)

model.fit(pad_train, y_train, batch_size = 64, validation_split = 0.1, epochs = 2)