import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer

def read_data(filename):
    file_content = pd.read_csv(filename)
    return file_content

train_data = read_data('C:/Users/rohan/Desktop/Dataset/Marathi_Train.csv')
test_data = read_data('C:/Users/rohan/Desktop/Dataset/Marathi_Test.csv')

X_train, X_valid, Y_train, Y_valid = train_data.Tweet, test_data.Tweet, train_data.Class, test_data.Class

tokenizer = Tokenizer(nb_words=2500, lower=True,split=' ')
tokenizer.fit_on_texts(train_data['Tweet'].values)
print(tokenizer.word_index)  # To see the dictionary