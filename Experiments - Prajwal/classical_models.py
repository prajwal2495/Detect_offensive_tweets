import pandas as pd
import numpy as np
import warnings

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
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


def train_test_TFIDF(train_data, test_data, model):
    print("TFIDF + ", model)
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 1)), model)
    X_train = train_data['tweet'].values.astype('U')
    y_train = train_data['subtask_a'].values.astype('U')

    X_test = test_data['tweet'].values.astype('U')
    y_test = test_data['subtask_a'].values.astype('U')
    model.fit(X_train, y_train)

    labels = model.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)

    cm = confusion_matrix(y_test, labels)
    print("Confusion matrix\n", cm)
    print(classification_report(y_test, labels, digits=4))
    print("\n\n")


def train_test_LDA(train_data, test_data, model):
    print("TFIDF + ", model)
    model = make_pipeline(CountVectorizer(), LDA(), model)
    X_train = train_data['tweet'].values.astype('U')
    y_train = train_data['subtask_b'].values.astype('U')

    X_test = test_data['tweet'].values.astype('U')
    y_test = test_data['subtask_b'].values.astype('U')
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)

    cm = confusion_matrix(y_test, labels)
    print("Confusion matrix\n", cm)
    print(classification_report(y_test, labels, digits=4))
    print("\n\n")


def train_test_BOW(train_data, test_data, model):
    print("TFIDF + ", model)
    model = make_pipeline(CountVectorizer(ngram_range=(1, 1)), model)
    X_train = train_data['tweet'].values.astype('U')
    y_train = train_data['subtask_b'].values.astype('U')

    X_test = test_data['tweet'].values.astype('U')
    y_test = test_data['subtask_b'].values.astype('U')
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, labels) * 100)

    cm = confusion_matrix(y_test, labels)
    print("Confusion matrix\n", cm)
    print(classification_report(y_test, labels, digits=4))

    print("\n\n")

def word_clouds(tweets):
    comment_words = ""
    map_of_words = {}
    for tweet in tweets:
        # comment_words += tweet + " "
        for word in str(tweet).split(" "):
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
        print("%s: %d" % (k, v))

    print()


def main():
    train_filename = './Data/MOLDV2_Train.csv'

    train_data = read_data(train_filename)
    #6987.88word_clouds(train_data[['tweet']])
    train_data = train_data[['tweet', 'subtask_a']]
    train_data = train_data[train_data['subtask_a'].notna()]

    test_filename = './Data/MOLDV2_Test.csv'
    test_data = read_data(test_filename)
    test_data = test_data[['tweet', 'subtask_a']]
    test_data = test_data[test_data['subtask_a'].notna()]

    #MNB = MultinomialNB()
    #SVC_obj = SVC()
    #DT = DecisionTreeClassifier()
    RF = RandomForestClassifier(random_state=42)
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 220, num=11)]
    #max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=RF, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)

    #train_test_BOW(train_data, test_data, MNB)
    #train_test_BOW(train_data, test_data, SVC_obj)
    train_test_TFIDF(train_data, test_data, rf_random)
    print(rf_random.best_params_)
    #train_test_BOW(train_data, test_data, DT)



if __name__ == '__main__':
    main()
