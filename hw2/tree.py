import graphviz
from sklearn import tree
from sklearn.metrics import accuracy_score
import pandas as pd
import pipeline

DLQ_COL = 'SeriousDlqin2yrs'
KEEP_COLS = ['SeriousDlqin2yrs', 'NumberRealEstateLoansOrLines']

filename = 'credit-data.csv'



class FinanceTree:
    '''
    Class for representing the symptom/diagnosis decision tree
    '''
    def __init__(self, data):
        self.model = tree.DecisionTreeClassifier()
        self.trained_model = None
        self.data = data
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.y_hat = None
        self.lookup = None


    def train(self, x_data, y_data):
        '''
        Trains the internal model using a set of x and y testing data
        Input:
            x_data (df): predictor variable data
            y_data (df): dependent variable data
        Output:
            Returns nothing
        '''
        self.x_train, self.x_test, \
        self.y_train, self.y_test = pipeline.get_test_train(x_data, y_data)
        self.trained_model = self.model.fit(self.x_train, self.y_train)


    def predict(self, param=None):
        '''
        Runs a prediciton on the trained model, either from the testing data
        or from a df with similar dimensions to the x_train data
        Input:
            param: either None to use the testing data, or a pandas df
            with the same cols as x_train and a single row of values
        Output:
            either None, if no param provided, or an int corresponding
            to the diagnosis
        '''
        if param is None:
            self.y_hat = self.trained_model.predict(self.x_test)
            return None
        if isinstance(param, pd.DataFrame):
            return self.trained_model.predict(param)

        return None


    @property
    def accuracy(self):
        '''
        Reports the accuracy of the trained model using testing data
        Output:
            Returns the trained SymptomTree class object
        '''
        return accuracy_score(self.y_test, self.y_hat)


def buildtree(raw_path):
    '''
    Main call to read data, create and train FinanceTree
    Input:
        raw_path (str): path of raw data csv
    Output:
        Returns the trained FinanceTree class object
    '''
    pd.options.mode.chained_assignment = None

    df = pipeline.load_data(raw_path)
    df = df.loc[:,KEEP_COLS]
    pipeline.fill_null(df)

    finance_tree = FinanceTree(df)

    x_train, y_train = pipeline.get_x_y_df( \
        finance_tree.data, [DLQ_COL])

    finance_tree.train(x_train, y_train)
    finance_tree.predict(None)

    return finance_tree


