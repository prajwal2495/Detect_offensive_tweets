import pandas as pd
import warnings

from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline


def read_data(filename):
    file_content = pd.read_csv(filename)
    return file_content

def word_extraction(sentence):
    words = sentence.split()
    cleaned_text = [w for w in words]
    return cleaned_text

def tokenize(sentences):
    words = []
    for sentence in sentences:
        w = word_extraction(sentence)
        words.extend(w)
        words = sorted(list(set(words)))
    return words

def TFIDF_PassiveAgressive(train_data, test_data):
    print("TFIDF + PassiveAgressive")
    model = make_pipeline(TfidfVectorizer(ngram_range=(1,1)), PassiveAggressiveClassifier(max_iter=100,random_state=0,tol=1e-3))
    X_train = train_data['Tweet']
    y_train = train_data['Class']

    X_test = test_data['Tweet']
    y_test = test_data['Class']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels, train_data['Class'].unique())
    print("Confusion matrix", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")

def main():
    train_filename = './Dataset/Marathi_Train.csv'
    train_data = read_data(train_filename)
    train_data = train_data[['Tweet', 'Class']]

    test_filename = './Dataset/Marathi_Test.csv'
    test_data = read_data(test_filename)
    test_data = test_data[['Tweet', 'Class']]

    TFIDF_PassiveAgressive(train_data, test_data)
