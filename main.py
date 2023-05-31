# This is a sample Python script.
import gensim
from typing import Callable, Optional, Sequence
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

from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('wordnet')
from collections import OrderedDict

import torch.nn as nn
import csv
import pandas as pd
import tensorflow as tf
import jieba
import transformers
import numpy as np
import train
# Define the list of descriptions
epsilon = 0.01
class classify():

    def __init__(self, file_in, num_clusters, is_english=False):
        self._file_path = file_in
        self._file_in = self._read_file(file_in)
        self._num_clusters = num_clusters
        self._is_english = is_english
        self.cumstom_lexicon()
        # Initialize the sentiment analyzer
        sia = SentimentIntensityAnalyzer()
        # update the sentiment lexicon
        sia.lexicon.update(self._custom_lexicon)
        self._sia = sia

    def _read_file(self,file_path):
        with open(file_path, 'r') as file:
            contents = file.read()
        return contents

    def parsing(self):
        data = []
        txt_file = self._file_in
        csv_file = 'preprocessesd.csv'
        lines = txt_file.split('\n')
        # Use a default-dict to group values by key
        from collections import defaultdict
        grouped_data = defaultdict(list)

        for line in lines:
            if '：' in line:
                parts = line.split('：')
                if len(parts[0]) >= 4:
                    key = parts[0][:4]
                else:
                    key = parts[0]
                value = parts[1]
                # Append the value to the list of values for the key
                grouped_data[key].append(value)
        # remove duplicate
        data_processed = [(key, ' '.join(values)) for key, values in grouped_data.items()]
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['stock', 'description'])
            for entry in data_processed:
                writer.writerow(entry)
        self._preprocessed = csv_file
        return csv_file

    # customize sentiment lexicon with biased words
    def cumstom_lexicon(self):
        custom_lexicon = {
            'improvement': 2.0,
            'highly recommend': 1.5,
            'not good': -1.0,
            'exceed expectations': 3,
            'cost is high': -10,
            'high growth': 5,
            'big increase': 1,
            'strong improvement': 10,
            'rapid growth': 1,
            'strong': 15
        }
        for word in list(custom_lexicon.keys()):
            sentiment_score = custom_lexicon[word]
            synonyms = []
            for synset in wordnet.synsets(word):
                for lemma in synset.lemmas():
                    synonyms.append(lemma.name())
            for synonym in synonyms:
                custom_lexicon[synonym] = sentiment_score
        self._custom_lexicon = custom_lexicon
        return custom_lexicon

    def score_calculator(self, text: str) -> dict[str, float]:
        return self._sia.polarity_scores(text)
    
    def sentiment_ana(self):
        file = self._preprocessed
        df = pd.read_csv(file)
        sentiment_data = df['description']
        scores = []  # Array to store computed scores
        senti_array = []

        for i, description in enumerate(sentiment_data):
            # if it's english, then no need to translate it
            if self._is_english:
                translator = Translator()
                description = translator.translate(description, timeout=20)
                # print(translated_text.text) #for debug use

            # Analyze the sentiment of the text
            sentiment_scores = self.score_calculator(description.text)
            # Store the data into the csv
            computed_score = self.compute_sentiment(sentiment_scores)
            scores.append(computed_score)
            senti_array.append(sentiment_scores)
            sentiment_data[i] = computed_score
            # Print the sentiment scores, for debug use
            # print(sentiment_scores)
            # print(computed_score)
            # for debug use
            if i > 50:
                break
            self._score = scores
            self._senti_array = senti_array
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
            print(x)
        self._sorted_array = sorted_array
        return sorted_array

    def cluster(self):
        array = self._sorted_array
        neg = []
        pos = []
        for stock in array:
            if stock[0] < 0 - epsilon:
                neg.append(stock[1])
            else:
                pos.append(stock[1])
        print("negative: ")
        print(neg)
        print("positive:")
        print(pos)



    # simple algorithm by multiplying compound and neg/pos/neu
    def compute_sentiment(self, senti_scores: dict[str, float]) -> float:
        if senti_scores['compound']>0:
            return senti_scores['pos']*senti_scores['compound'] - senti_scores['neu']* 0.05
        else:
            return senti_scores['neg'] * senti_scores['compound'] + senti_scores['neu'] * 0.05


    def train_nn(self):
        model = train.MLP(4, self._num_clusters)
        params = model.init({"params": np.ones((4,))})

        trainer = train.trainer('train.csv', model)
        # abstract four values to list/array
        input_tensor = []

        output_tensor = model.apply(params,input_tensor)

    #use nn to train
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
        self._padded= padded_sequences
        self._length = max_sequence_length
        return sequences

    def _autoencoder(self):
        input_dim = self._length
        encoding_dim =16
        df = pd.read_csv(self._preprocessed)
        input_data = tf.keras.layers.Input(shape=(input_dim,))
        encoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_data)
        decoder = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoder)
        autoencoder = tf.keras.models.Model(input_data, decoder)
        # use binary cross-entropy as loss function
        autoencoder.compile(loss='binary_crossentropy', optimizer='adam')
        autoencoder.fit(self._padded, self._padded, epochs=60, batch_size=16)
        encoder_model = tf.keras.models.Model(input_data,encoder)
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

    # # List of stock descriptions

    def run(self):
        self.parsing()
        self.tokenize()
        self._autoencoder()

    def debug(self):
        self.parsing()
        self.cumstom_lexicon()
        self.sentiment_ana()
        self.sort_result()
        self.cluster()
        # self._autoencoder()
        # print_csv(self._preprocessed)


def print_csv(file_path):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            print(row)


if __name__ == '__main__':
    model = train.MLP(4,3)
    trainer = train.trainer('train.csv',model)
    classifer = classify('咨询titles.txt',2)
    classifer.debug()
    # classifer.run()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
