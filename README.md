# Offensive Language Identification for Indo-Aryan Languages
### Contributors: Prajwal Krishna, Mayuresh Nene, Shrunali Paygude, Mrinal Chaudhari

## Overview of the project
- This project majorly focus on detecting the offensive tweets in a <b> low resource </b> language on twitter like Marathi.
- We are dealing with 3 levels of annotations.
   - Level A - Offensive or Not Offensive
   - Level B - Targeted insult (TIN) or Untargeted Insult (UNT)
   - Level C - Individual (IND), Group (Grp), or Others (OTH)

## Dataset
- The dataset is not one which was publicly available. We have developed a twitter scrapper which scans the entire twitter between given date range.
- After scanning the entire twitter, thanks to language specification argument which has been set to 'mr' meaning marathi, we get marathi tweets.
- After scrapping the twitter for tweets we have ended up with 13k tweets which were annotated manually and also woth the help of semi - supervised learning techniques.
- All the versions of dataset is available in the Data folder.

## Packages / Softwares required
- Python 2.7 or higher
- Jupyter notebook
- Any python IDE, Pycharm recommended
- Scikit-learn
- Tensorflow, Keras
- Pandas, numpy
- matplotlib, seaborn
- NLTK
- Scipy
- Pingouin

## How to run
- ### Classical models
  - Classical machine learning models like Decision Tree classifier, Random Forest Classifier, Support Vector Classifier, and Multinomial Naive bayes have been developed.
  - Executing them is pretty straightforward. make sure the datasets are in the directory they are intended to be in and click on run.
  - Each model has been tuned and will be tested on an unseen test dataset.
  - All the results, Confusion matrix and ROC plots that will be seen is on the unseen test dataset.

- ### Long Short Term Memory (LSTM)
  - LSTM is a nueral network model present in the tensorfow library
  - Make sure necessary libraries are included in the program 
  - Make sure your device has the required GPU to load the model, otherwise any environment or google collab should work fine.
  - This implmentation can be seen in the file LSTM_level_A,LSTM_level_B,LSTM_level_C python scripts.
  - Each level has an implementation because each level requires different dense layer adjustments.
  - I would recommend using the Jupyter notebook version of LSTM which is present in the LSTM directory under main directory
  - Using Jupyter notebook, the model summary, epoch training and ROC, precision recall curves are well plotted and represented.

The code is self explanatory and very easy to execute.

## Documentation 
- ### Doxygen
  - Doxygen was used to document all the python scripts.
  - All information on how each scripts works and the methods inside those scripts, what were they intended for is provided in this documentation.
  - The html folder is present inside the main directory. index.html file provides all the information.