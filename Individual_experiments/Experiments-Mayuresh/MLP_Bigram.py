#!/usr/bin/env python
# coding: utf-8

# In[5]:


# The path to the local git repo for Indic NLP library
INDIC_NLP_LIB_HOME=r"/Users/mayureshnene/Desktop/indic_nlp_library"

# The path to the local git repo for Indic NLP Resources
INDIC_NLP_RESOURCES=r"/Users/mayureshnene/Desktop/indic_nlp_resources"

import sys
import pandas as pd
import numpy as np
import warnings
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

from indicnlp import common
from indicnlp import loader
from indicnlp.tokenize import indic_tokenize
from indicnlp.tokenize import indic_tokenize 
from indicnlp.morph import unsupervised_morph 
from indicnlp import common

sys.path.append(r'{}'.format(INDIC_NLP_LIB_HOME))

common.set_resources_path(INDIC_NLP_RESOURCES)

loader.load()

training_dataset = pd.read_csv("/Users/mayureshnene/Desktop/MOLD/Mold/MOLDV2_Train.csv")
training_dataset.head()
training_dataset.dropna()
training_dataset['subtask_c'].fillna("NULL")

tweets = training_dataset["tweet"]
tweets

level_A = training_dataset[["subtask_a"]]
level_B = training_dataset.query("subtask_a == 'Offensive'")[["subtask_b"]]
level_C = training_dataset.query("subtask_b == 'TIN'")[["subtask_c"]]

stopwords_file = open("/Users/mayureshnene/Desktop/MOLD/Mold/stopwords.txt")
stopwords = stopwords_file.read().splitlines()
#print(stopwords)

def clean(tweet):
    removal_list = ['URL','\'ve','n\'t','\'s','\'m']
    for element in removal_list:
        tweet = str(tweet).replace(element,'')
    
    return tweet


iterator_map = map(clean,tweets)
tweets = list(iterator_map)

collective_tweets = copy.deepcopy(training_dataset)
analyzer=unsupervised_morph.UnsupervisedMorphAnalyzer('mr')

def tokenize(tweet):
    return indic_tokenize.trivial_tokenize(tweet)

def morph(tweet):
    analyzed_tokens=analyzer.morph_analyze_document(str(tweet).split(' '))
    return analyzed_tokens

tqdm.pandas(desc="Tokenize..")
#all_tweets["tokens"] = all_tweets['tweet'].progress_apply(tokenize)
collective_tweets["tokens"] = collective_tweets['tweet'].progress_apply(morph)

vector = collective_tweets["tokens"].tolist()

def ngram(vector):
	## Creates and stores an instance of the TfidfVectorizer class. This will be used further to extract our data as tf-idf features.
	vectorizer = CountVectorizer(analyzer = 'word', ngram_range = (2,2))
	untokenized_data =[' '.join(tweet) for tweet in tqdm(vector, "Vectorizing...")]
	vectorizer = vectorizer.fit(untokenized_data)
	vectors = vectorizer.transform(untokenized_data).toarray()
	return vectors

def get_vectors(vectors, labels, keyword):
	'''
	Returns a matrix for vectors. Zips vectors and labels IF and only if length of vector list is the same as length of the labels list. 
	Else, the function gets terminated.
	@param vectors These are the vectors for a given label.
	@param labels These are the label values for the given label.
	@param keyword which is the label to annotate for.
	'''
	if len(vectors) != len(labels):
		print("Unmatching sizes!")
		return
	
	## Stores a new list to append the zipped vectors and labels into.
	result = list()
	for vector, label in zip(vectors, labels):
		if label == keyword:
			result.append(vector)
	return result


vectors_level_A = ngram(vector)
labels_level_a = level_A['subtask_a'].values.tolist()
vectors_level_B = get_vectors(vectors_level_A, labels_level_a, "Offensive") 

labels_level_b = level_B['subtask_b'].values.tolist() 

## Numerical Vectors C
vectors_level_c = get_vectors(vectors_level_B, labels_level_b, "TIN") 

##Subtask C Labels
labels_level_c = level_C['subtask_c'].values.tolist() 

## Split into Train and Test vectors using the vectors of level A and Labels of level A with a training size of 0.75.
train_vectors_level_A, test_vectors_level_A, train_labels_level_A, test_labels_level_A = train_test_split(vectors_level_A[:], labels_level_a[:], train_size=0.70)

## Extracting names of labels and storing them in a variable
classNames = np.unique(test_labels_level_A)

print("Training begins on Level A classification...")
warnings.filterwarnings(action='ignore')

## Creating an object of SVC
classifiermlp = MLPClassifier(max_iter = 300)

#dt = DecisionTreeClassifier()

#dt_param_grid = [{'splitter':['best'], 'max_features': ['auto','100'], 'max_depth': [100,200,500,1000]}]

## Creating a parameter grid using the arguments SVC uses for hyper parameter tuning using GridSearchCV
param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}]

## Creating an object for GridSearchCV with the object of SVC and parameter grid given as parameters
# classifierGrid = GridSearchCV(classifiermlp, param_grid, refit = True, verbose=2)

#Model fit
classifiermlp.fit(train_vectors_level_A, train_labels_level_A)

# choosing the best estimator with the help of GridSearchCV
# classifierGrid = classifierGrid.best_estimator_
print("Training complete....")


print("calculating accuracy....")
## Training accuracy has been calculated
accuracy = accuracy_score(train_labels_level_A, classifiermlp.predict(train_vectors_level_A))
print("Training Accuracy:", accuracy)

## Predictions are obtained on the test data set
test_predictions = classifiermlp.predict(test_vectors_level_A)

## Testing accuracy has been calculated
accuracy = accuracy_score(test_labels_level_A, test_predictions)
print("Test Accuracy:", accuracy)


print("Confusion Matrix:")
## confusion matrix has been obtained for level A classification
matrix_level_A = confusion_matrix(test_labels_level_A, test_predictions)
print(matrix_level_A)
## Obtaining classification report for the test data set
print(classification_report(test_labels_level_A,test_predictions))

## plotting confusion matrix for better visualization
plottedCM_Level_A = plot_confusion_matrix(classifiermlp, test_vectors_level_A, test_labels_level_A,display_labels=classNames, cmap=plt.cm.Blues)
plt.show()


## Split into Train and Test vectors using the vectors of level A and Labels of level B with a training size of 0.75.
train_vectors_level_B, test_vectors_level_B, train_labels_level_B, test_labels_level_B = train_test_split(vectors_level_B[:], labels_level_b[:], train_size=0.75)

## Extracting names of labels and storing them in a variable
classNames = np.unique(test_labels_level_B)
print("Training begins on Level B classification...")
warnings.filterwarnings(action='ignore')

## Creating an object of SVC
classifiermlp = MLPClassifier(max_iter = 500)

#dt = DecisionTreeClassifier()

#dt_param_grid = [{'splitter':['best'], 'max_features': ['auto','100'], 'max_depth': [100,200,500,1000]}]

## Creating a parameter grid using the arguments SVC uses for hyper parameter tuning using GridSearchCV
#param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}]

## Creating an object for GridSearchCV with the object of SVC and parameter grid given as parameters
#classifierGrid = GridSearchCV(classifiermlp, param_grid, refit = True, verbose=2)

# Model fit
classifiermlp.fit(train_vectors_level_B, train_labels_level_B)

print("Training complete....")


print("calculating accuracy....")
## Training accuracy has been calculated
accuracy = accuracy_score(train_labels_level_B, classifiermlp.predict(train_vectors_level_B))
print("Training Accuracy:", accuracy)

## predictions are obtained on the testing data set
test_predictions = classifiermlp.predict(test_vectors_level_B)

## Testing accuracy has been calculated
accuracy = accuracy_score(test_labels_level_B, test_predictions)
print("Test Accuracy:", accuracy)

print("Confusion Matrix:")
## confusion matrix has been obtained for level A classification
matrix_level_B = confusion_matrix(test_labels_level_B, test_predictions)
print(matrix_level_B)
## Obtaining classification report for the test data set
print(classification_report(test_labels_level_B,test_predictions))

## Plotting confusion matrix for better visualization
plottedCM_Level_B = plot_confusion_matrix(classifiermlp, test_vectors_level_B, test_labels_level_B, display_labels=classNames, cmap=plt.cm.Blues)
plt.show()





## Split into Train and Test vectors using the vectors of level A and Labels of level B with a training size of 0.75.
train_vectors_level_C, test_vectors_level_C, train_labels_level_C, test_labels_level_C = train_test_split(vectors_level_c[:], labels_level_c[:], train_size=0.75)

## Extracting names of labels and storing them in a variable
classNames = np.unique(test_labels_level_C)
print("Training begins on Level C classification...")
warnings.filterwarnings(action='ignore')

## Creating an object of SVC
classifiermlp = MLPClassifier(max_iter = 500)



classifiermlp.fit(train_vectors_level_C, train_labels_level_C)

print("Training complete....")


print("calculating accuracy....")
## Training accuracy has been calculated
accuracy = accuracy_score(train_labels_level_C, classifiermlp.predict(train_vectors_level_C))
print("Training Accuracy:", accuracy)

## predictions are obtained on the testing data set
test_predictions = classifiermlp.predict(test_vectors_level_C)

## Testing accuracy has been calculated
accuracy = accuracy_score(test_labels_level_C, test_predictions)
print("Test Accuracy:", accuracy)


print("Confusion Matrix:")
## confusion matrix has been obtained for level A classification
matrix_level_C = confusion_matrix(test_labels_level_C, test_predictions)
print(matrix_level_C)
## Obtaining classification report for the test data set
print(classification_report(test_labels_level_C,test_predictions))

## plotting confusion matrix for better visualization
plottedCM = plot_confusion_matrix(classifiermlp, test_vectors_level_C, test_labels_level_C, display_labels=classNames, cmap=plt.cm.Blues)
plt.show()


# In[ ]:




