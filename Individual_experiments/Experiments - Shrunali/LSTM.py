import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             roc_auc_score)

def read_data(filename):
    file_content = pd.read_csv(filename)
    return file_content

train_data = read_data('C:/Users/rohan/Desktop/Dataset/Marathi_Train.csv')
test_data = read_data('C:/Users/rohan/Desktop/Dataset/Marathi_Test.csv')

X_train, X_valid, Y_train, Y_valid = train_data.Tweet, test_data.Tweet, train_data.Class, test_data.Class

tokenizer = Tokenizer(nb_words=2500, lower=True,split=' ')
tokenizer.fit_on_texts(train_data['Tweet'].values)
#print(tokenizer.word_index)  # To see the dictionary
X = tokenizer.texts_to_sequences(train_data['Tweet'].values)
X = pad_sequences(X)

Y = pd.get_dummies(train_data['Class']).values
X_valid = tokenizer.texts_to_sequences(test_data['Tweet'].values)
X_valid = pad_sequences(X_valid)
Y_valid = pd.get_dummies(test_data['Class']).values

embed_dim = 128
lstm_out = 256
batch_size = 16
model = Sequential()
model.add(Embedding(3000, embed_dim,input_length = X.shape[1]))
model.add(LSTM(lstm_out))
model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())



#Here we train the Network.

model.fit(X, Y, batch_size =batch_size, verbose = 5)
score, acc = model.evaluate(X_valid, Y_valid, verbose = 2, batch_size=batch_size)


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        print("recall====", recall)
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        print("precision= ", precision)
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    print("F1 score", 2 * ((precision * recall) / (precision + recall + K.epsilon())))
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

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

model.compile(loss='binary_crossentropy',optimizer= "adam",metrics=METRICS)
model.fit(X, Y, batch_size =batch_size, verbose = 5)
# score, acc = model.evaluate(X_valid, Y_valid, verbose = 2, batch_size=batch_size)

prediction = model.predict(X_valid)
y_pred = (prediction > 0.5)

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

report = classification_report(Y_valid, y_pred, digits=4)
print(report)
def plot_cm(labels, predictions, p=0.5):
    # print(labels)
    # print(predictions)
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion matrix (non-normalized))")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")


newLabels = []
for val in Y_valid:
    if val[0] == True and val[1] == False:
        newLabels.append(1)
    else:
        newLabels.append(0)

newPred = []
for val in y_pred:
    if val[0] == True and val[1] == False:
        newPred.append(1)
    else:
        newPred.append(0)
print(metrics.accuracy_score(newLabels, newPred)*100)
plot_cm(newLabels, newPred)