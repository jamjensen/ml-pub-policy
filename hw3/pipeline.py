import pandas as pd 
import numpy as np
import os
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.3
RANDOM_STATE = 1
DISCR_COL = 'MonthlyIncome'
BIN_LEN = 10000

KEEP_COLS = ['school_city','school_metro','school_charter','teacher_prefix', 
           'primary_focus_subject','primary_focus_area', 'secondary_focus_subject', 
           'secondary_focus_area','resource_type', 'poverty_level', 'grade_level',
           'eligible_double_your_impact_match', 'month_posted', 'year_posted']

TARGET_COL = ['outcome']

DATES = ['date_posted', 'datefullyfunded']

path = 'data/projects_2012_2013.csv'

class process:
    '''
    Class for data pipeline
    '''
    def __init__(self, path):

        self.data = self.load_data(path)
        self.cols = KEEP_COLS
        self.target = TARGET_COL
        self.discr_col = DISCR_COL
        self.y = None
        self.x = None
        # self.x = self.set_x()
        # self.y = self.set_y()
        # self.df = self.format_df()


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


def discretize(self, colname, bin_len):
    '''
    Discretizes a continuous variable
    Inputs:
        df: a dataframe
        colname (str): name of continuous variable
        bin_len (int): size of bins 
    '''
    
    lb = self.data[colname].min()
    ub = self.data[colname].max()
    bins = np.linspace(lb, ub, bin_len)

    df[colname +'_discrete'] = df[colname].apply(lambda x: np.digitize(x, bins))

    return None


def make_binary(df, cols):
    '''
    Makes a categorical variable binary
    Inputs:
        cols: list of column names
    '''

    df = pd.get_dummies(df, columns=cols)

    return df


def set_x(self):
    '''
    Creates predictor and target dataframes.
    Input:
        df (pandas df): containing predictor and target columns
        target_col (str): name of target column (predicting)
    Output:
        Returns a tuple of dataframes
    '''
    
    self.x = self.data.drop(self.target, axis=1)

    return None


def set_y(self):
    '''
    Creates predictor and target dataframes.
    Input:
        df (pandas df): containing predictor and target columns
        target_col (str): name of target column (predicting)
    Output:
        Returns a tuple of dataframes
    '''
    
    self.y = self.data.loc[:, target_col]

    return None


def format_df(df):
    '''
    Transforms feature columns into dummy variables and returns
        dataframe used to build the decision tree
    '''
    df = df.loc[:, DATES + KEEP_COLS + TARGET_COL]
    df2 = df.copy()
    rv = make_binary(df2, KEEP_COLS)

    return rv


def get_test_train(self):
    '''
    Splits data using scikit test_train
    Input:
        x: pandas df of predictor data
        y: pandas df of target data
    Output:
        x_train (df): dataframe
        x_test (df): dataframe
        y_train (df): dataframe
        y_test (df): dataframe
    '''
    x_train, x_test, y_train, y_test = train_test_split(self.x, self.y,
        test_size=TEST_SIZE, random_state=RANDOM_STATE)

    return x_train, x_test, y_train, y_test
