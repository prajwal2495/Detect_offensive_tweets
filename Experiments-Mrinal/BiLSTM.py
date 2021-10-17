from importlib import  reload
import sys
from imp import  reload

import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Concatenate, Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model, Sequential

def read_data(filename):
    file_content = pd.read_csv(filename)
    return file_content

train_data = read_data('./Data/Marathi_Train.csv')
test_data = read_data('./Data/Marathi_Test.csv')
train_data['Class'] = train_data['Class'].map({'Not Offensive': 1, 'Offensive': 0})
test_data['Class'] = test_data['Class'].map({'Not Offensive': 1, 'Offensive': 0})
X_train, X_valid, Y_train, Y_valid = train_data.Tweet, test_data.Tweet, train_data.Class, test_data.Class

type(Y_train[0])
EMBED_SIZE = 128
MAX_FEATURES = 3000

tokenizer = Tokenizer(num_words=MAX_FEATURES, lower=True,split=' ')
tokenizer.fit_on_texts(train_data['Tweet'].values)
list_tokenized_train = tokenizer.texts_to_sequences(train_data['Tweet'].values)

RNN_CELL_SIZE = 32
MAX_LEN = 300

X_train = pad_sequences(list_tokenized_train)

X_valid = tokenizer.texts_to_sequences(test_data['Tweet'].values)
X_valid = pad_sequences(X_valid)

sequence_input = Input(shape=(MAX_LEN,), dtype="int32")
embedded_sequences = Embedding(MAX_FEATURES, EMBED_SIZE)(sequence_input)

class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        score = tf.nn.tanh(
            self.W1(features) + self.W2(hidden_with_time_axis))
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights