##
# @mainpage Offensive Language Identification for Indo-Aryan Languages
#
# @section notes_main Overview
# - We have 13k tweets collected which are a mixture of Offensive and non offensive tweets
# - Count of offensive tweets : 2418
# - count of non offensive tweets : 10577
# - Along with Offensive and non Offensive tweets we do have a 2 more levels of annotations
#       - if the tweets are offensive they are sub-categorized into Targeted insult (TIN) and un-targeted insult (UNT)
#       - if the tweets are identified as TIN, they are sub-categorized into Individual (IND), Group(GRP), and Others (OTH).
# - The data was then pre-processed (training and testing) these pre-processed data was then fed to the model for training and testing was done on the unseen data.
# - Through statistical significance testing using pairwise T-tests and Mann-Whitney U we have found out that:
#       - Level A classification : LSTM is the best performing model.
#       - Level B classification : Random Forests Classifier and Support Vector classifier are the best performing models
#       - Level C classification : Random Forest Classifier is the best performing model.
#
# @section model_main Classical Model Training
# This project is mainly focused on the identifying the offensive tweets in low resource languages, we have chosen Marathi as our language of interest.
# Models like Random forest classifier, Decision tree classifier, Support vector classifier, Multinomial naive byes were used for training nad testing purposes.
# - Decision Tree Classifier
#    - Decision tree builds classification in the form of a tree structure while breaking the dataset in smaller and smaller subsets while incrementing the structure simultaneously. In our dataset, for level A, a decision tree would have to break the dataset in “Offensive” and “Non-Offensive”. Similarly for Level B and Level C, a decision tree would have the task to form a tree structure. With this background, a Decision Tree model was trained.
#    - Hyper-parameter tuning: The parameters used for tuning are max_depth, min_samples_leaf, criteria(gini, entropy).
#    - This was run for 3 cross validations and for 10 iterations each. These parameters were selected according to how the model performed on previous experiments.
# - Random Forest Classifier
#    - A Random Forest classifier estimates based on the combination of different Decision Trees. We can say that it fits a number of decision trees on various samples of the dataset. Each tree in the forest is built on random best set of features and this is what makes Random Forest one of the good performing algorithms while working with Natural Language Processing.
#    - For hyperparameter tuning, the max_features selected were “auto” and “sqrt”, max_depth was selected from a range, min_sample_split and min_sample_leaf were hardcoded and a grid was formed.
#    - Bootstrap (True/False) is used to select samples for training each tree
# - Multinomial Naive Bayes
#    - MNB has been known for performing better with text favoured tasks. Naive Bayes can be put as a simple text classification algorithm which is based on the probability of the events occurring such that there is no interdependence between the variables. One good feature of Naive Bayes is that it performs well with less training data as well, and in our case while we were working in phases of generating a low resource language dataset, we thought that it would perform good.
#    - Here, for hyperparameter tuning, we are using the alpha values ranging from 1 to 5 forming a grid to be tuned with 3 cross-validations and 100 iterations.
# - Support Vector Classifier
#    - SVM algorithm determines the best decision boundary between the vectors. It basically decides where to draw the best line that divides the space in distinctive subspaces.Our approach was to find vector representation which can encode as much information as possible and then we apply the SVM algorithm for classification.
#    - The class weight used for SVC is ‘balanced’
#    - The grid for SVC consists of kernel (linear, polynomial, rbf, sigmoid) and gamma as scale/auto.
#    - 3 cross validations with 100 iterations.
# @section LSTM_main LSTM Training
# - Why did we choose LSTM for the problem at hand
#   - Humans don't think from scratch, when we read an article, we understand each word based on the previous word. Traditional Neural networks fail to do this.
#   - Recurrent neural networks address this issue. They are networks with loops in them, allowing information to persist.
#   - Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. They were introduced by Hochreiter & Schmidhuber (1997).
#   - LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn! This exact behavior helps us tremendously as we are working on low resource languages and a model that remembers information/features for a long period of time is a necessity.
# - Model Summary:
#   - An embedding layer with parameters -
#       - Input dim = vocabulary size
#       - Output dim = 32
#       - Input length = size of the padded sequence
#       - Mask_zer0 = True to ignore 0
#   - An LSTM layer with parameter -
#       - Units = 100 (the resulting accuracy is almost same regardless of this value.
#   - Three dense layers
#   - An output dense layer with parameters
#       - Units = 2 and 3 for level A,B and C respectively.
#       - Activation = softmax ( for multi classification problem)
#   - Compilation with parameters
#       - Loss = categorical cross entropy
#       - Optimizer = adam
#       - Metrics = accuracy




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

def plot_cm(labels, predictions):
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
    ! This method uses the TFIDF vectorizer.
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
