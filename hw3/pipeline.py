import pandas as pd 
import numpy as np
import os
from sklearn.model_selection import train_test_split


FEATURES = ['school_state','school_metro','school_charter','teacher_prefix', 
            'students_reached','total_price_including_optional_support',
           'primary_focus_subject','primary_focus_area', 
           'resource_type', 'poverty_level', 'grade_level',
           'eligible_double_your_impact_match', 'month_posted', 'year_posted']

CONTINUOUS = ['total_price_including_optional_support', 'students_reached']

TARGET = ['not_funded_in_60']

DATES = ['date_posted', 'datefullyfunded']

path = 'data/projects_2012_2013.csv'

class process:
    '''
    Class for data pipeline
    '''
    def __init__(self, path):

        self.data = self.load_data(path)
        self.cols = FEATURES
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
    Inputs:
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


def make_binary(df, cols):
    '''
    Makes a categorical variable binary
    Inputs:
        cols: list of column names
    '''

    df = pd.get_dummies(df, columns=cols)

    return df



def format_df(df):
    '''
    Transforms feature columns into dummy variables and returns
        dataframe used to build the decision tree
    '''
    # df = df.loc[:, DATES + FEATURES + TARGET]
    df2 = df.copy()
    rv = make_binary(df2, FEATURES)

    return rv



def get_train_test_splits(df, train_start, train_end, test_start, test_end, continuous_cols):

    df_train = df[(df['date_posted'] >= train_start) & (df['date_posted'] <= train_end)]
    df_test = df[(df['date_posted'] >= test_start) & (df['date_posted'] <=test_end)]

    x_train = df_train[FEATURES]
    y_train = df_train[TARGET]

    x_test = df_test.loc[:,FEATURES]
    y_test = df_test.loc[:,TARGET]

    for df in [x_train, x_test]:
        fill_continuous_null(df, continuous_cols)
        # discretize(df, continuous_cols, 5)

    x_train['label'] = 'train'
    x_test['label'] = 'test'

    concat_df = pd.concat([x_train , x_test])

    features_df = make_binary(concat_df, FEATURES)

    x_train = features_df[features_df['label'] == 'train']
    x_test = features_df[features_df['label'] == 'test']

    x_train = x_train.drop('label', axis=1)
    x_test  = x_test.drop('label', axis=1)

    # x_train = make_binary(x_train, FEATURES)
    # x_test = make_binary(x_test, FEATURES)


    return x_train, y_train, x_test, y_test




