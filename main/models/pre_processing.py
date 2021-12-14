##
"""! @brief Defines the Pre-processing script"""


import re
import pandas as pd


# pre-processing the data
def clean_text(row, options):
    """! A Helps Cleaning all the individual tweets based on the selected options
    @param row  Individual tweets.
    @param options A dictionary containing True or False on several options pre-defined for pre-processing purposes.
    """

    if options['lowercase']:
        row = str(row).lower()

    if options['strip_spaces']:
        row = str(row).strip()

    if options['remove_url']:
        row = str(row).replace('http\S+|www.\S+', '')

    if options['remove_mentions']:
        row = re.sub("@[A-Za-z0-9]+","@USER",row)

    if options['remove_newline']:
        row = re.sub(r'\n',' ',row)

    if options['remove_tab']:
        row = re.sub(r'\t',' ',row)

    if options['remove_english']:
        row = re.sub("[A-Za-z0-9]+","",row)

    if options['add_USER_tag']:
        row = re.sub("@","@USER",row)

    if options['remove_specials']:
        row = re.sub('[+,-,_,=,/,<,>,!,#,$,%,^,&,*,\",:,;,.,' ',\t,\r,\n,\',|]','',row)
    return row

## clean_config is a dictionary that provides several options for pre-processing which helps to make the overall data noise free.
clean_config = {
    'remove_url': True,
    'remove_mentions': True,
    'decode_utf8': True,
    'lowercase': True,
    'remove_english': True,
    'remove_specials': True,
    'add_USER_tag': True,
    'remove_newline':True,
    'remove_tab':True,
    'strip_spaces':True
    }


def demoji(text):
    """! Helper function demojize is used remove the emojis from the tweets.
    Emojis does not help in understanding the data better in the perspective of teh model.
    @param text  Individual tweets.
    """
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U00010000-\U0010ffff"
                               "]+", flags=re.UNICODE)
    return(emoji_pattern.sub(r'', text))


def main(train):
    """! The trigger function main takes a parameter train to check whther train needs to pre-processed or test dataset.
    @param train Boolean value, True mean training dataset is being pre-processed, False means Testing dataset is being pre-processed
    """
    input_file = None
    if(train):
        input_file = 'Data/Training.csv'
    else:
        input_file = 'Data/Testing.csv'

    dataset = pd.read_csv(input_file)

    dataset_df = pd.DataFrame(dataset)

    dataset_df = dataset_df[["tweet","subtask_a", "subtask_b", "subtask_c"]]
    #, "subtask_a", "subtask_b", "subtask_c"

    ##lowe case conversion
    dataset_df['tweet'] = dataset_df['tweet'].str.lower()

    ##calling pre-processing function
    dataset_df['tweet'] = dataset_df['tweet'].apply(clean_text, args=(clean_config,))

    ##stripping leading and trailing whitespaces
    dataset_df['tweet'] = dataset_df['tweet'].str.strip()

    ##remove emojis - not working
    dataset_df.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))

    ##remove emojis - working
    dataset_df['tweet'] = dataset_df['tweet'].apply( lambda x : demoji(x))

    ##convert df to csv
    if(train):
        dataset_df.to_csv('./Data/Training_pre_processed.csv',index = False)
        return dataset_df
    else:
        dataset_df.to_csv('./Data/Training_pre_processed.csv',index = False)
        return dataset_df


if __name__ == "__main__":
    main(train=False)