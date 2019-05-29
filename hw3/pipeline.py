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



    # ### Make Target Column
    # df['diff'] = (df['datefullyfunded'] - df['date_posted']).dt.days
    # df['not_funded_in_60'] = np.where(df['diff'] > 60, 1, 0)



def get_train_test_splits(df, train_start, train_end, test_start, test_end, continuous_cols):


    str_columns = [column for column in df.columns if (df[column].dtype=='O') and (len(df[column].unique())<=51)]

    features_df = pd.get_dummies(df, columns=str_columns)

    for col in CONTINUOUS:
        features_df[col] = pd.qcut(df[col], 5)

    features_df['date_posted'] = df['date_posted']

    train_filter = (features_df['date_posted'] >= train_start) & (features_df['date_posted'] <= train_end)
    test_filter = (features_df['date_posted'] >= test_start) & (features_df['date_posted'] <=test_end)

    train_x, train_y = features_df[train_filter], df.TARGET[train_filter]
    test_x, test_y = features_df[test_filter], df.TARGET[test_filter]

    
    for df in [train_x, train_y]:
        fill_continuous_null(df, continuous_cols)


    return x_train, y_train, x_test, y_test




