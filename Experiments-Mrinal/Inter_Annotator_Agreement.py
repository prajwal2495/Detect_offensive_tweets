import pandas as pd

from sklearn.metrics import cohen_kappa_score

def main():
    files = ['./Inter_Annotator_Data/Mayuresh_mrinal.xlsx', './Inter_Annotator_Data/mrinal.xlsx']

    file_contents = []
    for file in files:
        file_content = pd.read_excel(file)
        file_contents.append(file_content)

        agreements = []
    for annotator in range(0, len(file_contents), 2):
        agreement = cohen_kappa_score((file_contents[annotator]['Class'][:100]), (file_contents[annotator + 1]['Class'][:100]))
        agreements.append(agreement)

        print(agreement)


if __name__ == '__main__':
    main()