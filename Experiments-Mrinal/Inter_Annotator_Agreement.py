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

        #print(agreement)
        #we need to calculate the percentage instead of printing agreement everytime.
    print("Average agreement: ", sum(agreements)/len(agreements))

def main2():
    files = ['./Inter_Annotator_Data/Mayuresh_mrinal.xlsx', './Inter_Annotator_Data/mrinal.xlsx']
    file_contents=pd.DataFrame()

    #What if duplicates?
    duplicate=0
    for file in files:
        file_content=pd.read_excel(file)
        if duplicate%2==0:
            file_contents.append(file_content)
        else:
            file_contents.append(file_content[100:]) #append after 100th
        duplicate+=1
    # print(len(file_contents))
    # print(file_contents['Class'].unique())
    print(len(file_contents[file_contents['Class'] == 'offensive']))
    print(len(file_contents[file_contents['Class'] == 'not offensive']))


if __name__ == '__main__':
    main()