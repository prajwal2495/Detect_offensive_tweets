import pandas as pd

def read_data(filename):
    file_content = pd.read_csv(filename)
    return file_content

train_data = read_data('./Data/Marathi_Train.csv')
test_data = read_data('./Data/Marathi_Test.csv')
train_data['Class'] = train_data['Class'].map({'Not Offensive': 1, 'Offensive': 0})
test_data['Class'] = test_data['Class'].map({'Not Offensive': 1, 'Offensive': 0})
X_train, X_valid, Y_train, Y_valid = train_data.Tweet, test_data.Tweet, train_data.Class, test_data.Class

type(Y_train[0])

MAX_FEATURES = 3000

tokenizer = Tokenizer(num_words=MAX_FEATURES, lower=True,split=' ')
tokenizer.fit_on_texts(train_data['Tweet'].values)
list_tokenized_train = tokenizer.texts_to_sequences(train_data['Tweet'].values)