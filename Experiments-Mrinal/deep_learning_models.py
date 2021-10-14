import pandas as pd
import warnings
from keras.layers import Embedding, Dense, Activation
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from keras.models import Sequential
warnings.filterwarnings("ignore")

def read_data(filename):
    file_content = pd.read_csv(filename)
    return file_content


def TFIDF_Multi_Layer_Perceptron(train_data, test_data):
    model = make_pipeline(TfidfVectorizer(),MLPClassifier(random_state=1, max_iter=100, learning_rate_init=0.001, activation='logistic'))
    X_train = train_data['Tweet']
    y_train = train_data['Class']

    X_test = test_data['Tweet']
    y_test = test_data['Class']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    cm = confusion_matrix(y_test, labels, train_data['Class'].unique())
    print("Confusion matrix", cm)


def main():
    train_filename = './Dataset/Marathi_Train.csv'
    train_data = read_data(train_filename)
    train_data = train_data[['Tweet', 'Class']]

    test_filename = './Dataset/Marathi_Test.csv'
    test_data = read_data(test_filename)
    test_data = test_data[['Tweet', 'Class']]


if __name__ == '__main__':
    main()
