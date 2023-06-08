import gensim
from typing import Callable, Optional, Sequence, Tuple
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from googletrans import Translator
import nltk
from nltk.corpus import wordnet
import train
import meta_parameters
from nltk.sentiment import SentimentIntensityAnalyzer


from collections import OrderedDict

import torch.nn as nn

import pandas as pd
import tensorflow as tf
import jieba
import transformers
import numpy as np
import helper
import meta_parameters
from deep_translator import GoogleTranslator



class predictor():

    def __init__(self, file_in, trainer_in, num_clusters=3, is_english=False, mode='simple'):
        self._file_in = helper._read_file(file_in)
        self._num_clusters = num_clusters
        self._is_english = is_english
        self.custom_lexicon()
        self._sia = meta_parameters.sia
        # set to simple algorithm by default
        self._mode = mode
        # define the trained model
        self._model = trainer_in


    def parsing(self):
        self._preprocessed = helper.parsing(self._file_in, mode='pred')
        return self._preprocessed

    # customize sentiment lexicon with biased words
    def custom_lexicon(self):
        self._custom_lexicon_ = helper._custom_lexicon_fn()

    def score_calculator(self, text: str) -> dict[str, float]:
        return helper._score_calculator(self._sia, text)

    def group(self,dict_in):
        pos = dict_in['pos']
        neg = dict_in['neg']
        if abs(pos-neg ) < 0.1:
            self._dic['ambiguous'].append()

    def sentiment_ana(self):
        file = self._preprocessed
        df = pd.read_csv(file)
        sentiment_data = df['description']
        stocks = df['stock']
        scores = []  # Array to store computed scores
        senti_array = []
        dict = {'neg': [], 'pos': [], 'ambiguous': [], 'pos+': [], 'neu': []}
        self._dic = dict
        for i, description in enumerate(sentiment_data):
            translator = GoogleTranslator(source='auto', target='en')
            if len(description) > 100:
                description = description[:100]
            text0 = translator.translate(description)
            # Analyze the sentiment of the text
            sentiment_scores = self.score_calculator(text0)

            pos = sentiment_scores['pos']
            neg = sentiment_scores['neg']
            neu = sentiment_scores['neu']
            if abs(pos - neg) < 0.1 and (pos > 0.1 or neg > 0.1):
                dict['ambiguous'].append(stocks.loc[i])
            elif neg > 0.1:
                dict['neg'].append(stocks.loc[i])
            elif pos > 0.5:
                dict['pos+'].append(stocks.loc[i])
            elif neu > 0.8 and (pos < 0.1 or neg < 0.1):
                dict['neu'].append(stocks.loc[i])
            elif pos > 0.1:
                dict['pos'].append(stocks.loc[i])
            elif pos > 0.1 and neg > 0.1:
                dict['ambiguous'].append(stocks.loc[i])
            else:
                dict['ambiguous'].append(stocks.loc[i])
            # Store the data into the csv
            computed_score = self.compute_sentiment(sentiment_scores,)
            scores.append(computed_score)
            senti_array.append(sentiment_scores)
            sentiment_data[i] = computed_score
            # Print the sentiment scores, for debug use
            print(sentiment_scores)
            print(stocks.loc[i])
            # print(computed_score)
            # for debug use
            if i > 50:
                break
            self._score = scores
            self._senti_array = senti_array
        print(dict)
        return

    def sort_result(self):
        file = self._preprocessed
        df = pd.read_csv(file)
        stocks = df['stock']
        score = self._score
        labled_array = []
        dict_array = self._senti_array
        for i, stock in enumerate(stocks):
            if i >= len(score):
                break

            labled_array.append((score[i], stock, dict_array[i]))
            # print(score[i])
            # print(stock)

        sorted_array = sorted(labled_array,
                              key=lambda x: x[0])
        for x in sorted_array:
            # print(x)
            print(x)
        self._sorted_array = sorted_array
        return sorted_array

    def cluster(self):
        array = self._sorted_array
        neg = []
        pos = []
        for stock in array:
            if stock[0] < 0 - meta_parameters.epsilon:
                neg.append(stock[1])
            else:
                pos.append(stock[1])
        print("negative: ")
        print(neg)
        print("positive:")
        print(pos)

    # simple algorithm by multiplying compound and neg/pos/neu
    def compute_sentiment(self, senti_scores: dict[str, float]) -> float:
        if self._mode == 'simple':
            # pos = senti_scores['pos']
            # neg = senti_scores['neg']
            # neu = senti_scores['neu']
            # comp = senti_scores['compound']
            # result = pos*pos+neg*neg
            if senti_scores['compound'] > 0:
                return senti_scores['pos'] * senti_scores['compound'] - senti_scores['neu'] * 0.05
            else:
                return senti_scores['neg'] * senti_scores['compound'] + senti_scores['neu'] * 0.05
        # by far the other choice is mlp neural network
        else:
            y_pred = self._model(list(senti_scores.values()))
            print("print compute_sentiment")
            print(y_pred)
            return y_pred

    def tokenize(self):
        file = self._preprocessed
        df = pd.read_csv(file)
        tokenized_descriptions = df['description']
        for i, description in enumerate(tokenized_descriptions):
            tokens = jieba.lcut(description)
            tokenized_descriptions[i] = ' '.join(tokens)
            # tokenized_descriptions._append(' '.join(tokens))

        df['tokenized_description'] = tokenized_descriptions
        print(tokenized_descriptions)
        # Save the updated DataFrame to a new CSV file
        df.to_csv('tokenized_file.csv', index=True)
        self._tokenized = df
        # Create a vocabulary
        vocab = set([token for description in tokenized_descriptions for token in description])
        vocab_size = len(vocab)

        # Convert the descriptions to sequences of tokens
        sequences = []
        for description in tokenized_descriptions:
            sequence = [list(vocab).index(token) for token in description]
            sequences.append(sequence)

        # Pad sequences to the same length
        max_sequence_length = max(len(sequence) for sequence in sequences)
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=max_sequence_length, padding='post'
        )
        self._padded = padded_sequences
        self._length = max_sequence_length
        return sequences

    def _autoencoder(self):
        input_dim = self._length
        encoding_dim = 16
        df = pd.read_csv(self._preprocessed)
        input_data = tf.keras.layers.Input(shape=(input_dim,))
        encoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_data)
        decoder = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoder)
        autoencoder = tf.keras.models.Model(input_data, decoder)
        # use binary cross-entropy as loss function
        autoencoder.compile(loss='binary_crossentropy', optimizer='adam')
        autoencoder.fit(self._padded, self._padded, epochs=60, batch_size=16)
        encoder_model = tf.keras.models.Model(input_data, encoder)
        embeddings = encoder_model.predict(self._padded)

        # Apply clustering algorithm,try K-means here
        from sklearn.cluster import KMeans

        kmeans = KMeans(self._num_clusters)
        kmeans.fit(embeddings)
        cluster_labels = kmeans.labels_

        # print(df['description'])
        # Print the cluster labels

        for description, label in zip(df['description'], cluster_labels):
            print(f"{description} - Cluster {label}")

    def run(self):
        self.parsing()
        self.tokenize()
        self._autoencoder()

    def debug(self):
        self.parsing()
        # self.custom_lexicon()
        self.sentiment_ana()
        # self.sort_result()
        # self.sort_result()
        # self.cluster()
        # self._autoencoder()
        # print_csv(self._preprocessed)
