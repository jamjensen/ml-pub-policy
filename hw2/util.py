import data_pipeline as dp
import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2
RANDOM_STATE = 1


def get_df_from_csv(path, keep_cols):
    '''
    Reads and processes the dataset, including encoding the dependent variable
    numerically. Uses one-hot-encoding to transform symptoms and demographic
    characteristics into dummy variables. Returns the updated dataframe,
    a dictionary of diagnosis codes mapped to diagnosis strings, and
    a dictionary containing the same information with values as keys.
    Inputs:
        path (str): csv file
        keep_cols (lst): list of columns to include in the dataframe
        d_col (str): column name for original target column
        d_cat_col (str): column name for new target column
        dummies (lst): column names for the new dummy variables
        prefx_cols (lst): list of prefixes of the new column names
    Outputs:
        df2 (dataframe): updated dataframe
        d_map (dict): dictionary mapping diagnosis strings to a unique index
        d_r (dict): dictionary mapping unique numbers to diagnosis strings
    '''
    df = dp.read_and_process_data(path)
    df = df.loc[:, keep_cols]
    df2 = df.copy()
    df2 = dp.make_binary(df, cols)

    return df2


