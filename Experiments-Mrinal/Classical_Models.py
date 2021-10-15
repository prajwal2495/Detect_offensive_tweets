import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def read_data(filename):
    file_content = pd.read_csv(filename)
    return file_content

def TFIDF_KNN(train_data, test_data):
    print("TFIDF + KNN")
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 1)), KNeighborsClassifier(n_neighbors=5))
    X_train = train_data['Tweet']
    y_train = train_data['Class']

    X_test = test_data['Tweet']
    y_test = test_data['Class']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)


def main():
    train_filename = './Data/Marathi_Train.csv'
    train_data = read_data(train_filename)
    train_data = train_data[['Tweet', 'Class']]

    test_filename = './Data/Marathi_Test.csv'
    test_data = read_data(test_filename)
    test_data = test_data[['Tweet', 'Class']]

    TFIDF_KNN(train_data, test_data)