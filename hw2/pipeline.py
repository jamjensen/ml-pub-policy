import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2
RANDOM_STATE = 1

filename = 'credit-data.csv'

def load_data(filename):

	df = pd.read_csv(filename, index_col=False)
	
	return df

## Explore data



## Pre-Process 

def fill_null(df):
	'''
	Fills null columns
	'''
	if any(df.isna()): 
		df.fillna(df.median(), inplace=True)

	return None


def discretize(df, colname, bin_len):
	'''
	Discretizes a continuous variable
	How should we choose number of bins? 
	'''
	lb = df[colname].min()
	ub = df[colname].max()
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


def get_x_y_df(df, target_col):
    '''
    Creates predictor and target dataframes.
    Input:
        df (pandas df): containing predictor and target columns
        target_col (str): name of target column (predicting)
    Output:
        Returns a tuple of dataframes
    '''
    x = df.drop(target_col, axis=1)
    y = df.loc[:, target_col]

    return x, y



def get_test_train(x, y):
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
    x_train, x_test, y_train, y_test = train_test_split(x, y,
        test_size=TEST_SIZE, random_state=RANDOM_STATE)

    return x_train, x_test, y_train, y_test
