# generics
import pandas as pd
import pre_processing
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# visu
import seaborn as sns
import matplotlib.pyplot as plt

# texts
import re
import unicodedata
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model

# Model
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping

# NLTK
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn import metrics


def read_data(filename):
    """! A file reader.
    @param filename  The name of the Training or Testing dataset.
    @return  A Pandas dataframe of the given dataset
    """
    file_content = pd.read_csv(filename)
    return file_content


def feature_target_preparation(train_df, train_df_encoded, test_df, test_df_encoded):
    y_train = train_df['subtask_a'].copy()
    y_test = test_df['subtask_a'].copy()

    y_train_encoded = to_categorical(train_df_encoded['subtask_a'], 2)
    y_test_encoded = to_categorical(test_df_encoded['subtask_a'], 2)

    y_train_mapped = train_df_encoded['subtask_a'].copy()
    y_test_mapped = test_df_encoded['subtask_a'].copy()

    X_train = train_df_encoded[['tweet']].copy()
    X_test = test_df_encoded[['tweet']].copy()
    return y_train, y_test, y_train_encoded, y_test_encoded, y_train_mapped, y_test_mapped, X_train, X_test


def get_maxlen_tweet(X_train):
    max_word_count = 0
    word_count = []
    #
    for encoded_tweet in X_train:
        word_count.append(len(encoded_tweet))
        if len(encoded_tweet) > max_word_count:
            max_word_count = len(encoded_tweet)
    print("Maximum number of word in one tweet: " + str(max_word_count) + " words")
    return max_word_count


def build_LSTM_model(vocab_length, max_len_tweet):
    model_LSTM = Sequential()
    model_LSTM.add(layers.Embedding(vocab_length, output_dim=32, input_length=max_len_tweet, mask_zero=True))
    model_LSTM.add(layers.LSTM(100))
    model_LSTM.add(layers.Dense(64, activation="relu"))
    model_LSTM.add(layers.Dense(32, activation="relu"))
    model_LSTM.add(layers.Dense(16, activation="relu"))
    model_LSTM.add(layers.Dense(2, activation='softmax'))
    model_LSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model_LSTM.summary())
    return model_LSTM


def fit_LSTM_MODEL(LSTM_MODEL, X_train, y_train_encoded, X_test, y_test_encoded):
    es = EarlyStopping(patience=10, monitor='val_accuracy', restore_best_weights=True)
    history = LSTM_MODEL.fit(X_train,
                             y_train_encoded,
                             validation_data=(X_test, y_test_encoded),
                             epochs=30,
                             batch_size=16,
                             verbose=1,
                             callbacks=[es]
                             )
    return history


def final_report(LSTM_MODEL, X_test, y_test_mapped, y_test):
    predicted = LSTM_MODEL.predict(X_test)
    y_pred = predicted.argmax(axis=-1)
    acc_score = accuracy_score(y_test_mapped, y_pred)

    report = classification_report(y_test_mapped, y_pred, target_names=list(y_test.unique()), output_dict=True)
    accuracy_col = ([""] * 3) + [round(acc_score, 2)]
    accuracy_col = pd.Series(accuracy_col, index=list(report["Offensive"].keys()))

    df_report = pd.DataFrame(report)[["Offensive", "not offensive", "macro avg", "weighted avg"]].apply(
        lambda x: round(x, 2))
    df_report["accuracy"] = accuracy_col
    return df_report, y_pred, predicted


def plot_cm_ROC(y_test_mapped, y_pred, y_test, y_train, predicted):
    cm = confusion_matrix(y_test_mapped, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot()
    plt.show()
    y_test_array = pd.get_dummies(y_test_mapped, drop_first=False).values
    classes = y_train.unique()

    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(18.5, 5)
    ## Plot roc
    for i in range(len(classes)):
        fpr, tpr, thresholds = roc_curve(y_test_array[:, i], predicted[:, i])
        ax[0].plot(fpr, tpr, lw=3,
                   label='{0} (area (AUC) = {1:0.2f})'.format(classes[i], metrics.auc(fpr, tpr))
                   )
    ax[0].plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    ax[0].set(xlim=[-0.05, 1.0], ylim=[0.0, 1.05],
              xlabel='False Positive Rate',
              ylabel="True Positive Rate (Recall)",
              title="Receiver operating characteristic")
    ax[0].legend(loc="lower right")
    ax[0].grid(True)

    ## Plot precision-recall curve
    for i in range(len(classes)):
        precision, recall, thresholds = metrics.precision_recall_curve(
            y_test_array[:, i], predicted[:, i])
        ax[1].plot(recall, precision, lw=3,
                   label='{0} (area ={1:0.2f})'.format(classes[i], metrics.auc(recall, precision))
                   )
    ax[1].set(xlim=[0.0, 1.05], ylim=[0.0, 1.05], xlabel='Recall',
              ylabel="Precision", title="Precision-Recall curve")
    ax[1].legend(loc="best")
    ax[1].grid(True)
    plt.show()


def main():

    train_df = pre_processing.main(train=True)
    test_df = pre_processing.main(train=False)

    train_df_encoded = train_df.copy()
    test_df_encoded = test_df.copy()
    map_sentiment = {"Offensive": 0, "not offensive": 1}
    train_df['subtask_a'] = train_df_encoded['subtask_a'].map(map_sentiment)
    test_df['subtask_a'] = test_df_encoded['subtask_a'].map(map_sentiment)

    y_train, y_test, y_train_encoded, y_test_encoded, y_train_mapped, y_test_mapped, X_train, X_test = feature_target_preparation(
        train_df, train_df_encoded, test_df, test_df_encoded)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train['tweet'])
    vocab_length = len(tokenizer.word_index) + 1

    X_train = tokenizer.texts_to_sequences(X_train['tweet'])
    X_test = tokenizer.texts_to_sequences(X_test['tweet'])

    max_len_tweet = get_maxlen_tweet(X_train)

    X_train = pad_sequences(X_train, maxlen=max_len_tweet, padding='post')
    X_test = pad_sequences(X_test, maxlen=max_len_tweet, padding='post')

    LSTM_MODEL = build_LSTM_model(vocab_length, max_len_tweet)

    LSTM_MODEL_history = fit_LSTM_MODEL(LSTM_MODEL, X_train, y_train_encoded, X_test, y_test_encoded)

    df_report, y_pred, predicted = final_report(LSTM_MODEL, X_test, y_test_mapped, y_test)

    plot_cm_ROC(y_test_mapped, y_pred, y_test, y_train, predicted)

if __name__ == '__main__':
    main()
