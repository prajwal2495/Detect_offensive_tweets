import pandas as pd

def read_data(filename):
    file_content = pd.read_csv(filename)
    return file_content
