import pandas as pd 
import numpy as np
import os
from sklearn.model_selection import train_test_split


CONTINUOUS = ['total_price_including_optional_support', 'students_reached']

TARGET = 'not_funded_in_60'

DATES = ['date_posted', 'datefullyfunded']

path = 'data/projects_2012_2013.csv'

class process:
    '''
    Class for data pipeline
    '''
    def __init__(self, path):

        self.data = self.load_data(path)
        self.target = TARGET

    def load_data(self, path):
        '''
        The data pipeline
        Inputs:
            filename (str): name of csv
        Outputs:
            df: a dataframe
        '''

        if os.path.exists(path):
            df = pd.read_csv(path)
        else:
            raise Exception('The file does not exist')
        
        return df


def fill_continuous_null(df, cols):
    '''
    Fills null columns in a dataframe in place
    Inputs:str_columns
        cols: list of column names corresponding to continuous variables
            with null values
    '''

    for col in cols:
        df[col].fillna(df[col].median(), inplace=True)

    return None


def discretize(df, cols, num_bins):
    '''
    Discretizes a continuous variable
    Inputs:
        df: a dataframe
        colname (str): name of continuous variable
        bin_len (int): size of bins 
    '''
    
    for col in cols:
        df[col] = pd.qcut(df[col], num_bins)

    # df[colname +'_discrete'] = df[colname].apply(lambda x: np.digitize(x, bins))

    return df



    # ### Make Target Column
    # df['diff'] = (df['datefullyfunded'] - df['date_posted']).dt.days
    # df['not_funded_in_60'] = np.where(df['diff'] > 60, 1, 0)



def get_train_test_splits(df, train_start, train_end, test_start, test_end):
    '''
    Given start and end dates for both training and testing datasets, this function identifies
    and makes dummy variables out of categorical features, imputes missing data for continuous
    variables only using data in either the test or training dataset, and returns the testing and
    training datasets to be used each model.
    '''

    str_columns = [column for column in df.columns if (df[column].dtype=='O') and (len(df[column].unique())<=51)]

    features_df = pd.get_dummies(df[str_columns], dummy_na=True, columns=str_columns)

    for col in CONTINUOUS:
        features_df[col] = df[col]

    train_filter = (df['date_posted'] >= train_start) & (df['date_posted'] <= train_end)
    test_filter = (df['date_posted'] >= test_start) & (df['date_posted'] <=test_end)

    train_x, train_y = features_df[train_filter], df[train_filter][TARGET]
    test_x, test_y = features_df[test_filter], df[test_filter][TARGET]

    
    for dframe in [train_x, test_x]:
        fill_continuous_null(dframe, CONTINUOUS)


    return train_x, train_y, test_x, test_y




