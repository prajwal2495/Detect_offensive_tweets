import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import LatentDirichletAllocation as LDA
import warnings
warnings.filterwarnings("ignore")

def read_data(filename):
    file_content = pd.read_csv(filename)
    return file_content

def TFIDF_KNN(train_data, test_data):
    print("TFIDF + KNN")
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 1)), KNeighborsClassifier(n_neighbors=5))
    X_train = train_data['tweet'].values.astype('U')
    y_train = train_data['subtask_a'].values.astype('U')

    X_test = test_data['tweet'].values.astype('U')
    y_test = test_data['subtask_a'].values.astype('U')
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels)
    print("Confusion matrix\n", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")


def TFIDF_GradientBoosting(train_data, test_data):
    print("TFIDF + GradientBoost")
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 1)), GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0))
    X_train = train_data['tweet'].values.astype('U')
    y_train = train_data['subtask_a'].values.astype('U')

    X_test = test_data['tweet'].values.astype('U')
    y_test = test_data['subtask_a'].values.astype('U')
    model.fit(X_train, y_train)
    labels = model.predict(X_test)

    # takes a lot of time to generate 10 trees and find accuracy
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    cm = confusion_matrix(y_test, labels)
    print("Confusion matrix", cm)
    print(classification_report(y_test, labels, digits=4))
    print("\n\n")

def BOW_KNN(train_data, test_data):
    print("BOW + KNN")
    model = make_pipeline(CountVectorizer(ngram_range=(1, 1)), KNeighborsClassifier(n_neighbors=5))
    X_train = train_data['tweet'].values.astype('U')
    y_train = train_data['subtask_a'].values.astype('U')

    X_test = test_data['tweet'].values.astype('U')
    y_test = test_data['subtask_a'].values.astype('U')
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels, train_data['subtask_a'].unique())
    print("Confusion matrix", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")

def LDA_KNN(train_data, test_data):
    print("LDA + KNN")
    model = make_pipeline(CountVectorizer(), LDA(), KNeighborsClassifier(n_neighbors=5))
    X_train = train_data['tweet'].values.astype('U')
    y_train = train_data['subtask_a'].values.astype('U')

    X_test = test_data['tweet'].values.astype('U')
    y_test = test_data['subtask_a'].values.astype('U')
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels, train_data['subtask_a'].unique())
    print("Confusion matrix", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")

def main():
    train_filename = 'Data/MOLDV2_Train.csv'
    train_data = read_data(train_filename)
    train_data = train_data[['tweet', 'subtask_a']]
    train_data = train_data[train_data['subtask_a'].notna()]


    test_filename = './Data/MOLDV2_Test.csv'
    test_data = read_data(test_filename)
    test_data = test_data[['tweet', 'subtask_a']]
    test_data = test_data[test_data['subtask_a'].notna()]

    TFIDF_KNN(train_data, test_data)
    TFIDF_GradientBoosting(train_data, test_data)
    BOW_KNN(train_data, test_data)
    LDA_KNN(train_data, test_data)


if __name__ == '__main__':
    main()
