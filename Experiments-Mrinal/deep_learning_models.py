import pandas as pd
import warnings
from keras.layers import Embedding, Dense, Activation
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report,  roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from keras.models import Sequential
warnings.filterwarnings("ignore")

def read_data(filename):
    file_content = pd.read_csv(filename)
    return file_content


def TFIDF_Multi_Layer_Perceptron(train_data, test_data):
    print("TFIDF + MultiLayer Perceptron")
    model = make_pipeline(TfidfVectorizer(),MLPClassifier(random_state=1, max_iter=100, learning_rate_init=0.001, activation='logistic'))
    X_train = train_data['Tweet']
    y_train = train_data['Class']

    X_test = test_data['Tweet']
    y_test = test_data['Class']

    model.fit(X_train, y_train)
    labels = model.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    cm = confusion_matrix(y_test, labels)
    print("Confusion matrix", cm)

    print(classification_report(y_test, labels, digits=4))
    print("\n\n")

    # for j in range(len(y_test)):
    #     p.append(1 - decision[j][0])
    # print("AUC_SCORE:", roc_auc_score(y_test, p))
    # print("\n\n")


def main():
    train_filename = './Data/Marathi_Train_level_C.csv'
    train_data = read_data(train_filename)
    train_data = train_data[['Tweet', 'Class']]
    train_data = train_data[train_data['Class'].notna()]

    test_filename = './Data/Marathi_Test_Level_C.csv'
    test_data = read_data(test_filename)
    test_data = test_data[['Tweet', 'Class']]
    test_data = test_data[test_data['Class'].notna()]

    TFIDF_Multi_Layer_Perceptron(train_data, test_data)

if __name__ == '__main__':
    main()
