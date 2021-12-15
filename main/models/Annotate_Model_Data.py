#!/usr/bin/env python
# coding: utf-8

# In[18]:


# The path to the local git repo for Indic NLP library
INDIC_NLP_LIB_HOME=r"/Data/indic_nlp_library"

# The path to the local git repo for Indic NLP Resources
INDIC_NLP_RESOURCES=r"Data/indic_nlp_resources"

import sys
import pandas as pd
import numpy as np
import warnings
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

from indicnlp import common
from indicnlp import loader
from indicnlp.tokenize import indic_tokenize
from indicnlp.tokenize import indic_tokenize 
from indicnlp.morph import unsupervised_morph 
from indicnlp import common

sys.path.append(r'{}'.format(INDIC_NLP_LIB_HOME))

common.set_resources_path(INDIC_NLP_RESOURCES)

loader.load()


# In[19]:


training_dataset = pd.read_csv("../Fully_Annotated_cleaned.csv")
#training_dataset.dropna()
training_dataset.fillna("NULL")


# In[20]:


tweets = training_dataset["tweet"]
tweets


# In[21]:


level_A = training_dataset[["subtask_a"]]
# level_B = training_dataset.query("subtask_a == 'Offensive'")[["subtask_b"]]
# level_C = training_dataset.query("subtask_b == 'TIN'")[["subtask_c"]]
level_B_indices = training_dataset.index[training_dataset['subtask_a'] == "Offensive"].tolist()
level_C_indices = training_dataset.index[training_dataset['subtask_b'] == "TIN"].tolist()


# In[23]:


len(level_C_indices)


# In[24]:


level_B = training_dataset["subtask_b"][level_B_indices]
level_C = training_dataset["subtask_c"][level_C_indices]
level_C


# In[ ]:





# In[25]:


stopwords_file = open("/Users/mayureshnene/Desktop/MOLD/Mold/stopwords.txt")
stopwords = stopwords_file.read().splitlines()
print(stopwords)


# In[26]:


def clean(row):
    row = str(row)
#     removal_list = ['URL','\'ve','n\'t','\'s','\'m','!']
#     for element in removal_list:
#         row = row.replace(element,'')
    
    row = row.replace('http\S+|www.\S+', '')
    row = re.sub("@[A-Za-z0-9]+","@USER",row)
    row = re.sub("[A-Za-z0-9]+","",row)
    row = re.sub("@","@USER",row)
    row = re.sub('[+,-,_,=,/,<,>,!,#,$,%,^,&,*,\",:,;,.,' ',\t,\r,\n,\',|]','',row)
    return row


# In[27]:


# for tweet in tweets:
#     tweet = remove_noise(str(tweet))
iterator_map = map(clean,tweets)
tweets = list(iterator_map)


# In[28]:




collective_tweets = copy.deepcopy(training_dataset)


# In[29]:




analyzer=unsupervised_morph.UnsupervisedMorphAnalyzer('mr')


# In[30]:


def tokenize(tweet):
    return indic_tokenize.trivial_tokenize(tweet)

def morph(tweet):
    analyzed_tokens=analyzer.morph_analyze_document(str(tweet).split(' '))
    return analyzed_tokens


# In[ ]:





# In[31]:


tqdm.pandas(desc="Tokenize..")
#all_tweets["tokens"] = all_tweets['tweet'].progress_apply(tokenize)
collective_tweets["tokens"] = collective_tweets['tweet'].progress_apply(morph)


# In[ ]:





# In[32]:


vector = collective_tweets["tokens"].tolist()


# In[33]:


def tfid_vectorizer(vector):
	## Creates and stores an instance of the TfidfVectorizer class. This will be used further to extract our data as tf-idf features.
	vectorizer = TfidfVectorizer()
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


# In[34]:


vectors_level_A = tfid_vectorizer(vector)
labels_level_a = level_A['subtask_a'].values.tolist()
vectors_level_B = get_vectors(vectors_level_A, labels_level_a, "Offensive") 

vectors_level_B

labels_level_b = level_B.values.tolist() 

## Numerical Vectors C
vectors_level_c = get_vectors(vectors_level_B, labels_level_b, "TIN") 

##Subtask C Labels
labels_level_c = level_C.values.tolist() 


# In[58]:


# train_vectors_level_A, train_labels_level_A,= vectors_level_A[1:3097], labels_level_a[1:3097]
# test_vectors_level_A, test_labels_level_A = vectors_level_A[3097:], labels_level_a[3097:]
# ## Extracting names of labels and storing them in a variable
# classNames = np.unique(test_labels_level_A)

# print("Training begins on Level A classification...")
# warnings.filterwarnings(action='ignore')

# ## Creating an object of SVC, MNB, SGD, MLP
# classifiersvc = SVC()
# classifiermnb = MultinomialNB()
# classifiersgd = SGDClassifier()
# classifiermlp = MLPClassifier()


# ## Fit on Level A
# classifiersvc.fit(train_vectors_level_A, train_labels_level_A)
# classifiermnb.fit(train_vectors_level_A, train_labels_level_A)
# classifiersgd.fit(train_vectors_level_A, train_labels_level_A)
# classifiermlp.fit(train_vectors_level_A, train_labels_level_A)


# ## Predict on Level A
# test_predictions_A_svc = classifiersvc.predict(test_vectors_level_A)
# test_predictions_A_mnb = classifiermnb.predict(test_vectors_level_A)
# test_predictions_A_sgd = classifiersgd.predict(test_vectors_level_A)
# test_predictions_A_mlp = classifiermlp.predict(test_vectors_level_A)

# preds_A = pd.DataFrame(columns = ['SVC_A', 'MNB_A', 'SGD_A','MLP_A'])
# label_name = ['SVC_A', 'MNB_A','SGD_A','MLP_A']
# preds = [test_predictions_A_svc, test_predictions_A_mnb, test_predictions_A_sgd, test_predictions_A_mlp]

# i = 0
# for label in label_name:
#     preds_A[label] = preds[i]
#     i += 1


# preds_A.to_csv("../data/Level_A_Annotated.csv")


# In[35]:


## Split into Train and Test vectors using the vectors of level A and Labels of level B with a training size of 0.75.

# train_vectors_level_B, train_labels_level_B,= vectors_level_B[1:1066], labels_level_b[1:1066]
# test_vectors_level_B, test_labels_level_B = vectors_level_B[1066:2420], labels_level_b[1066:2420]

# ## Extracting names of labels and storing them in a variable
# classNames = np.unique(test_labels_level_B)
# print("Training begins on Level B classification...")
# warnings.filterwarnings(action='ignore')

# ## Creating an object of SVC, MNB, SGD, MLP
# classifiersvc = SVC()
# classifiermnb = MultinomialNB()
# classifiersgd = SGDClassifier()
# classifiermlp = MLPClassifier()

# ## Fit on Level B
# classifiersvc.fit(train_vectors_level_B, train_labels_level_B)
# classifiermnb.fit(train_vectors_level_B, train_labels_level_B)
# classifiersgd.fit(train_vectors_level_B, train_labels_level_B)
# classifiermlp.fit(train_vectors_level_B, train_labels_level_B)

# ## Predict on Level B
# test_predictions_B_svc = classifiersvc.predict(test_vectors_level_B)
# test_predictions_B_mnb = classifiermnb.predict(test_vectors_level_B)
# test_predictions_B_sgd = classifiersgd.predict(test_vectors_level_B)
# test_predictions_B_mlp = classifiermlp.predict(test_vectors_level_B)


# ## Split into Train and Test vectors using the vectors of level A and Labels of level B with a training size of 0.75.
train_vectors_level_C, train_labels_level_C = vectors_level_c[1:739], labels_level_c[1:739]
test_vectors_level_C, test_labels_level_C = vectors_level_c[739:], labels_level_c[739:]

## Extracting names of labels and storing them in a variable
classNames = np.unique(test_labels_level_C)
print("Training begins on Level C classification...")
warnings.filterwarnings(action='ignore')

## Creating an object of SVC, MNB, SGD, MLP
classifiersvc = SVC()
classifiermnb = MultinomialNB()
classifiersgd = SGDClassifier()
classifiermlp = MLPClassifier()

## Fit on Level C
classifiersvc.fit(train_vectors_level_C, train_labels_level_C)
classifiermnb.fit(train_vectors_level_C, train_labels_level_C)
classifiersgd.fit(train_vectors_level_C, train_labels_level_C)
classifiermlp.fit(train_vectors_level_C, train_labels_level_C)



## Predict on Level C
test_predictions_C_svc = classifiersvc.predict(test_vectors_level_C)
test_predictions_C_mnb = classifiermnb.predict(test_vectors_level_C)
test_predictions_C_sgd = classifiersgd.predict(test_vectors_level_C)
test_predictions_C_mlp = classifiermlp.predict(test_vectors_level_C)


# In[60]:


preds_B = pd.DataFrame(columns = ['SVC_B', 'MNB_B', 'SGD_B','MLP_B'])
label_name = ['SVC_B', 'MNB_B','SGD_B','MLP_B']
preds = [test_predictions_B_svc, test_predictions_B_mnb, test_predictions_B_sgd, test_predictions_B_mlp]

i = 0
for label in label_name:
    preds_B[label] = preds[i]
    i += 1
    
preds_B.to_csv("../data/Level_B_Annotated.csv")


# In[ ]:





# In[36]:


preds_C = pd.DataFrame(columns = ['SVC_C', 'MNB_C', 'SGD_C','MLP_C'])
label_name = ['SVC_C', 'MNB_C','SGD_C','MLP_C']
preds = [test_predictions_C_svc, test_predictions_C_mnb, test_predictions_C_sgd, test_predictions_C_mlp]

i = 0
for label in label_name:
    preds_C[label] = preds[i]
    i += 1

preds_C.to_csv("../data/Level_C_Annotated.csv")


# In[62]:


#final_df = pd.DataFrame()
final_df = pd.concat([preds_A, preds_B, preds_C], ignore_index=True)


# In[ ]:


final_df


# In[ ]:


#final_df.to_csv("/Users/mayureshnene/Desktop/Mayuresh/Offense_Marathi/Experiments/Data_annotated.csv")


# In[ ]:




