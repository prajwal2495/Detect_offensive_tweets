import pandas as pd

def read_data(filename):
    file_content = pd.read_csv(filename)
    return file_content

def main():
    train_filename = './Dataset/Marathi_Train.csv'
    train_data = read_data(train_filename)
    train_data = train_data[['Tweet', 'Class']]

    test_filename = './Dataset/Marathi_Test.csv'
    test_data = read_data(test_filename)
    test_data = test_data[['Tweet', 'Class']]
    
if __name__ == '__main__':
    main()