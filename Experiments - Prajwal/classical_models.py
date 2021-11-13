import pandas as pd
import numpy as np
import warnings

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report, roc_curve,auc,roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
#import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import LatentDirichletAllocation as LDA

warnings.filterwarnings("ignore")
from sklearn.svm import SVC

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
def TFIDF_KNN(train_data, test_data):
    print("TFIDF + KNN")
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 1)), KNeighborsClassifier(n_neighbors=5))
    X_train = train_data['tweet']
    y_train = train_data['subtask_a']

    X_test = test_data['tweet']
    y_test = test_data['subtask_a']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels)#, train_data['Class'].unique())
    print("Confusion matrix\n", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")

def TFIDF_Decision(train_data, test_data):
    print("TFIDF + Decision tree")
    model = make_pipeline(TfidfVectorizer(ngram_range=(1,1)), DecisionTreeClassifier())
    X_train = train_data['tweet']
    y_train = train_data['subtask_a']

    X_test = test_data['tweet']
    y_test = test_data['subtask_a']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels)#, train_data['Class'].unique())
    print("Confusion matrix\n", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")


def TFIDF_Random_forest(train_data, test_data):
    print("TFIDF + Random forest")
    model = make_pipeline(TfidfVectorizer(ngram_range=(1,1)), RandomForestClassifier())
    X_train = train_data['tweet']
    y_train = train_data['subtask_a']

    X_test = test_data['tweet']
    y_test = test_data['subtask_a']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels)#, train_data['Class'].unique())
    print("Confusion matrix\n", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")

def TFIDF_SVM(train_data, test_data):
    print("TFIDF + SVM")
    model = make_pipeline(TfidfVectorizer(ngram_range=(1,1)), SVC())
    X_train = train_data['tweet']
    y_train = train_data['subtask_a']

    X_test = test_data['tweet']
    y_test = test_data['subtask_a']
    model.fit(X_train, y_train)
    #AUC_ARRAY = []
    #alpha = [10 ** x for x in range(-5, 4)]
    labels = model.predict(X_test)
    #calib = CalibratedClassifierCV(model, method="sigmoid")
    #calib.fit(X_train,y_train)
    #predictions = calib.predict_proba(X_test)
    #AUC_ARRAY.append(roc_auc_score(y_test,predictions[:,1]))
   # for i in range(len(AUC_ARRAY)):
    #    print('AUC for alpha = ', alpha[i], 'is', AUC_ARRAY[i])

    #best_alpha = np.argmax(AUC_ARRAY)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels)#, train_data['Class'].unique())
    print("Confusion matrix\n", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")

def TFIDF_Multi_Naive_Bayes(train_data, test_data):
    print("TFIDF + MultiNomial Naive Bayes")
    model = make_pipeline(TfidfVectorizer(ngram_range=(1,1)), MultinomialNB())
    X_train = train_data['tweet']
    y_train = train_data['subtask_a']

    X_test = test_data['tweet']
    y_test = test_data['subtask_a']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels)#, train_data['Class'].unique())
    print("Confusion matrix\n", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")

def BOW_Random_forest(train_data, test_data):
    print("BOW + Random forest")
    model = make_pipeline(CountVectorizer(ngram_range=(1, 1)), RandomForestClassifier())
    X_train = train_data['tweet']
    y_train = train_data['subtask_a']

    X_test = test_data['tweet']
    y_test = test_data['subtask_a']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels)#, train_data['Class'].unique())
    print("Confusion matrix\n", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")


def BOW_Decision_Tree(train_data, test_data):
    print("BOW + Decsion Tree")
    model = make_pipeline(CountVectorizer(ngram_range=(1, 1)), DecisionTreeClassifier())
    X_train = train_data['tweet']
    y_train = train_data['subtask_a']

    X_test = test_data['tweet']
    y_test = test_data['subtask_a']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels)#, train_data['Class'].unique())
    print("Confusion matrix\n", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")


def BOW_Multi_Naive_Bayes(train_data, test_data):
    print("BOW + Naive Bayes")
    model = make_pipeline(CountVectorizer(ngram_range=(1, 1)), MultinomialNB())
    X_train = train_data['tweet']
    y_train = train_data['subtask_a']

    X_test = test_data['tweet']
    y_test = test_data['subtask_a']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels)#, train_data['Class'].unique())
    print("Confusion matrix\n", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")


def BOW_SVM(train_data, test_data):
    print("BOW + SVM")
    model = make_pipeline(CountVectorizer(ngram_range=(1,1)), SVC())
    # model = make_pipeline(TfidfVectorizer(), SVC())
    X_train = train_data['tweet']
    y_train = train_data['subtask_a']

    X_test = test_data['tweet']
    y_test = test_data['subtask_a']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels)#, train_data['Class'].unique())
    print("Confusion matrix\n", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")

def LDA_SVM(train_data, test_data):
    print("LDA + SVM")
    model = make_pipeline(CountVectorizer(), LDA(), SVC())
    X_train = train_data['tweet']
    y_train = train_data['subtask_a']

    X_test = test_data['tweet']
    y_test = test_data['subtask_a']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels)#, train_data['Class'].unique())
    print("Confusion matrix\n", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")

def LDA_Random_forest(train_data, test_data):
    print("LDA + Random forest")
    model = make_pipeline(CountVectorizer(), LDA(), RandomForestClassifier())
    X_train = train_data['tweet']
    y_train = train_data['subtask_a']

    X_test = test_data['tweet']
    y_test = test_data['subtask_a']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels)#, train_data['Class'].unique())
    print("Confusion matrix\n", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")


def LDA_Multi_Naive_Bayes(train_data, test_data):
    print("LDA + MultiNomial Naive Bayes")
    model = make_pipeline(CountVectorizer(), LDA(), MultinomialNB())
    X_train = train_data['tweet']
    y_train = train_data['subtask_a']

    X_test = test_data['tweet']
    y_test = test_data['subtask_a']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels)#, train_data['Class'].unique())
    print("Confusion matrix\n", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")


def LDA_Decision(train_data, test_data):
    print("LDA + Decision tree")
    model = make_pipeline(CountVectorizer(), LDA(), DecisionTreeClassifier())
    X_train = train_data['tweet']
    y_train = train_data['subtask_a']

    X_test = test_data['tweet']
    y_test = test_data['subtask_a']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)
    # takes a lot of time to generate 10 trees and find accuracy
    # print(cross_val_score(model, X, y, cv=10))
    cm = confusion_matrix(y_test, labels)#, train_data['Class'].unique())
    print("Confusion matrix\n", cm)
    print(classification_report(y_test, labels, digits = 4))
    print("\n\n")


def word_clouds(tweets):
    comment_words = ""
    map_of_words = {}
    for tweet in str(tweets['tweet']):
        # comment_words += tweet + " "
        for word in tweet.split():
            if word in map_of_words:
                map_of_words[word] += 1
            else:
                map_of_words[word] = 1
    map_of_words = [(v, k) for k, v in map_of_words.items()]
    map_of_words.sort(reverse=True)  # natively sort tuples by first element
    i = 0
    for v, k in map_of_words:
        if i == 5:
            break
        i += 1
        print ("%s: %d" % (k, v))

    print()




def main():
    train_filename = 'Data/OLD_DATA/MOLDV2_Train.csv'

    train_data = read_data(train_filename)
    #word_clouds(train_data[['tweet']])
    train_data = train_data[['tweet', 'subtask_a']]
    train_data = train_data[train_data['subtask_a'].notna()]

    # x_train, y_train, x_test, y_test = train_test_split(train_data['Tweet'],train_data['Class'],test_size=0.3,random_state=42)
    # # print(x_train)
    # x_train.dropna()
    # y_test.dropna()
    # x_test.dropna()
    # y_test.dropna()
    
    test_filename = './Data/MOLDV2_Test.csv'
    test_data = read_data(test_filename)
    test_data = test_data[['tweet', 'subtask_a']]
    test_data = test_data[test_data['subtask_a'].notna()]

    # print(len(train_data[train_data['subtask_a'] == 'offensive']))
    # print(len(train_data[train_data['subtask_a'] == 'not offensive']))
    #
    # print(len(test_data[test_data['subtask_a'] == 'offensive']))
    # print(len(test_data[test_data['subtask_a'] == 'not offensive']))
    #
    # print(len(train_data))
    # print(len(test_data))
    TFIDF_KNN(train_data, test_data)
    # TFIDF_Decision(train_data, test_data)
    # TFIDF_Multi_Naive_Bayes(train_data, test_data)
    # TFIDF_Random_forest(train_data, test_data)
    # TFIDF_SVM(train_data, test_data)
    #
    # BOW_Decision_Tree(train_data, test_data)
    # BOW_Multi_Naive_Bayes(train_data, test_data)
    # BOW_Random_forest(train_data, test_data)
    # BOW_SVM(train_data, test_data)
    #
    # LDA_Decision(train_data, test_data)
    # LDA_Multi_Naive_Bayes(train_data, test_data)
    # LDA_Random_forest(train_data, test_data)
    # LDA_SVM(train_data, test_data)


if __name__ == '__main__':
    main()
