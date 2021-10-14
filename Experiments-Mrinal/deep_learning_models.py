import pandas as pd


def read_data(filename):
    file_content = pd.read_csv(filename)
    return file_content


def TFIDF_Multi_Layer_Perceptron(train_data, test_data):
    model = make_pipeline(TfidfVectorizer(),MLPClassifier(random_state=1, max_iter=100, learning_rate_init=0.001, activation='logistic'))
    X_train = train_data['Tweet']
    y_train = train_data['Class']

    X_test = test_data['Tweet']
    y_test = test_data['Class']
    model.fit(X_train, y_train)
    labels = model.predict(X_test)

def main():
    train_filename = './Dataset/Marathi_Train.csv'
    train_data = read_data(train_filename)
    train_data = train_data[['Tweet', 'Class']]

    test_filename = './Dataset/Marathi_Test.csv'
    test_data = read_data(test_filename)
    test_data = test_data[['Tweet', 'Class']]


if __name__ == '__main__':
    main()
