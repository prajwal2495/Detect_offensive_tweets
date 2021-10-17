import pandas as pd
import warnings

from sklearn import metrics
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

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
