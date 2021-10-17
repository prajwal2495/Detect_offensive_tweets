from importlib import reload
import sys
from imp import reload

import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Concatenate, Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model, Sequential
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             roc_auc_score)
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

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

lstm = Bidirectional(LSTM(RNN_CELL_SIZE, return_sequences = True), name="bi_lstm_0")(embedded_sequences)
# Getting our LSTM outputs
(lstm, forward_h, forward_c, backward_h, backward_c) = Bidirectional(LSTM(RNN_CELL_SIZE, return_sequences=True, return_state=True), name="bi_lstm_1")(lstm)

state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
context_vector, attention_weights = Attention(10)(lstm, state_h)
dense1 = Dense(20, activation="relu")(context_vector)
dropout = Dropout(0.05)(dense1)
output = Dense(1, activation="sigmoid")(dropout)

model = keras.Model(inputs=sequence_input, outputs=output)
print(model.summary())
keras.utils.plot_model(model, show_shapes=True, dpi=90)

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
]

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=METRICS)

print(X_train.shape, Y_train.shape)
BATCH_SIZE = 100
EPOCHS = 3
history = model.fit(X_train,Y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_split=0.2)
prediction = model.predict(X_valid)
y_pred = (prediction > 0.5)
report = classification_report(Y_valid, y_pred, digits=4)
#print(report)

def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion matrix (non-normalized))")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")


plot_cm(Y_valid, y_pred)
