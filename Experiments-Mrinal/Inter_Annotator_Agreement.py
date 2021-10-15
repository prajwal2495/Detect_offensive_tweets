import pandas as pd

from sklearn.metrics import cohen_kappa_score

def main():
    files = ['./Inter_Annotator_Data/Mayuresh_mrinal.xlsx', './Inter_Annotator_Data/mrinal.xlsx']

    file_contents = []
    for file in files:
        file_content = pd.read_excel(file)
        file_contents.append(file_content)

if __name__ == '__main__':
    main()