
#!/usr/bin/env python3
"""! @brief A python script implementing machine learning algorithms for sentiment analysis"""

import pandas as pd
import numpy as np
import warnings
import pre_processing

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")



def read_data(filename):
    """! A file reader.
    @param filename  The name of the Training or Testing dataset.
    @return  A Pandas dataframe of the given dataset
    """
    file_content = pd.read_csv(filename)
    return file_content


def word_extraction(sentence):
    """! A word extractor.
    @param sentence  The name of the Training or Testing dataset.
    """
    words = sentence.split()
    cleaned_text = [w for w in words]
    return cleaned_text


def tokenize(sentences):
    """! word tokenizer methodr.
    @param sentences Each tweet in the dataset.
    """
    words = []
    for sentence in sentences:
        w = word_extraction(sentence)
        words.extend(w)
        words = sorted(list(set(words)))
    return words

def plot_cm(labels, predictions, p=0.5):
    """! A confusion matrix plotter.
    @param labels The ground truth labels of tweets
    @param predictions the predictions of the machine learning models
    """
    # print(labels)
    # print(predictions)
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion matrix (non-normalized))")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.show()



def train_test_TFIDF(train_data, test_data, model):
    """! This method uses the training data to train a given model and test on the unseen testing dataset.
    ! This method uses the TFIDF vectorizer
    ! This method also plots the confusion matrix and the ROC AUC curve of a given model.
    @param train_data The name of the Training dataset.
    @param test_data The name of the Testing dataset.
    @param model object of the machine learning model.
    """
    print("TFIDF + ", model)
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 5)), model)

    X_train = train_data['tweet'].values.astype('U')
    y_train = train_data['subtask_a'].values.astype('U')

    X_test = test_data['tweet'].values.astype('U')
    y_test = test_data['subtask_a'].values.astype('U')

    model.fit(X_train, y_train)

    labels = model.predict(X_test)

    print("Testing Accuracy:", metrics.accuracy_score(y_test, labels) * 100)

    cm = confusion_matrix(y_test, labels)
    print("Confusion matrix\n", cm)
    print(classification_report(y_test, labels, digits=4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot()
    plt.show()
    ax = plt.gca()
    roc_disp = plot_roc_curve(model,X_test,y_test ,ax=ax,alpha=0.8)
    plt.show()

    print("\n\n")


def train_test_LDA(train_data, test_data, model):
    """! This method uses the training data to train a given model and test on the unseen testing dataset.
    ! This method uses the Latent Dirichlet allocation
    ! This method also plots the confusion matrix and the ROC AUC curve of a given model.
    @param train_data The name of the Training dataset.
    @param test_data The name of the Testing dataset.
    @param model object of the machine learning model.
    """
    print("TFIDF + ", model)
    model = make_pipeline(CountVectorizer(), LDA(), model)
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

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot()
    plt.show()
    ax = plt.gca()
    roc_disp = plot_roc_curve(model,X_test,y_test ,ax=ax,alpha=0.8)
    plt.show()

    print("\n\n")


def train_test_BOW(train_data, test_data, model):
    """! This method uses the training data to train a given model and test on the unseen testing dataset.
    ! This method uses the Count Vectorizer
    ! This method also plots the confusion matrix and the ROC AUC curve of a given model.
    @param train_data The name of the Training dataset.
    @param test_data The name of the Testing dataset.
    @param model object of the machine learning model.
    """
    print("TFIDF + ", model)
    model = make_pipeline(CountVectorizer(ngram_range=(1, 1)), model)
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

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot()
    plt.show()
    ax = plt.gca()
    roc_disp = plot_roc_curve(model,X_test,y_test ,ax=ax,alpha=0.8)
    plt.show()

    print("\n\n")

def word_clouds(tweets):
    """! A Word cloud generating method. This method does not plot a traditional word cloud instead gives a dictionary of words and it's frequencies.
    @param tweets Each tweet in the dataset
    """
    comment_words = ""
    map_of_words = {}
    for tweet in tweets:
        # comment_words += tweet + " "
        for word in str(tweet).split(" "):
            if word in map_of_words and word is not None and word != "" :
                map_of_words[word] += 1
            else:
                map_of_words[word] = 1
    map_of_words = [(v, k) for k, v in map_of_words.items()]
    map_of_words.sort(reverse=True)  # natively sort tuples by first element
    i = 0
    for v, k in map_of_words:
        if i == 100:
            break
        i += 1
        print("%s: %d" % (k, v))

    # with open('weights_words_100.csv','w') as file:
    #     for value,key in map_of_words:
    #         file.write("%s,%s\n"%(key,value))

    print()


def main():
    train_df = pre_processing.main(train=True) #'Data/Training_pre_processed.csv'

    train_data = train_df#read_data(train_filename)
    #word_clouds(train_data['tweet'])
    train_data = train_data[['tweet', 'subtask_a']]
    train_data = train_data[train_data['subtask_a'].notna()]

    test_df = pre_processing.main(train=False)#'Data/Testing_pre_processed.csv'
    test_data = test_df
    test_data = test_data[['tweet', 'subtask_a']]
    test_data = test_data[test_data['subtask_a'].notna()]

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 220, num=11)]
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
    RF = RandomForestClassifier(random_state=42)
    rf_random = RandomizedSearchCV(estimator=RF, param_distributions=random_grid, n_iter=10, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)

    # train_test_TFIDF(train_data, test_data, rf_random)
    # print(rf_random.best_params_)
    # print("End of RFC")


    DT = DecisionTreeClassifier()
    DT_grid = {'max_depth':max_depth,
               'min_samples_leaf':[5,10,20,50,100],
               'criterion':['gini','entropy']}
    dt_random = RandomizedSearchCV(estimator=DT,param_distributions=DT_grid,n_iter=10,cv=3,verbose=2,
                                   random_state=42, n_jobs=-1)

    train_test_TFIDF(train_data, test_data, dt_random)
    print(dt_random.best_params_)
    print("end of DTC")


    MNB = MultinomialNB(fit_prior=True)
    MNB_grid = {'alpha':[1.0,2.0,3.0,4.0,5.0]
                }
    MNB_random = RandomizedSearchCV(estimator=MNB,param_distributions=MNB_grid,n_iter=100,cv=3,verbose=2,
                                   random_state=42, n_jobs=-1)

    # train_test_TFIDF(train_data, test_data, MNB_random)
    # print(MNB_random.best_params_)
    # print("End of MNB")

    SVC_obj = SVC(random_state=42,class_weight='balanced')
    SVC_Grid = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma':['scale','auto']
                }
    SVC_random = RandomizedSearchCV(estimator=SVC_obj,param_distributions=SVC_Grid,n_iter=100,cv=3,verbose=2,
                                   random_state=42, n_jobs=-1)

    # train_test_TFIDF(train_data, test_data, SVC_random)
    # print(SVC_random.best_params_)
    # print("End of SVC")


if __name__ == '__main__':
    main()
