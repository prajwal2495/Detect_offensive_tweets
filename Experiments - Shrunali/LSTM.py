import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

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