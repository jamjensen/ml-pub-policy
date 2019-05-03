
from __future__ import division
import pandas as pd
import numpy as np
from sklearn import preprocessing, svm, metrics, tree, decomposition, svm
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import random
import matplotlib.pyplot as plt
from scipy import optimize
import time
import seaborn as sns
import pipeline



def define_clfs_params(grid_size):
    """Define defaults for different classifiers.
    Define three types of grids:
    Test: for testing your code
    Small: small grid
    Large: Larger grid that has a lot more parameter sweeps
    """

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        # 'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'DT': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'BG': BaggingClassifier(LogisticRegression(penalty='l1', C=1e5))
            }

    large_grid = { 
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
    'BG': {'n_estimators' : [10, 20], 'max_samples' : [.25, .5]}
           }

    
    test_grid = { 
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'LR': { 'penalty': ['l1'], 'C': [0.01]},
    'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
    'DT': {'criterion': ['gini'], 'max_depth': [1],'min_samples_split': [10]},
    # 'SVM' :{'C' :[0.01],'kernel':['linear']},
    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']},
    'BG': {'n_estimators' : [10], 'max_samples' : [.5]}
           }

    if (grid_size == 'large'):
        return clfs, large_grid
    elif (grid_size == 'test'):
        return clfs, test_grid
    else:
        return None



def run_time_loop(df, models_to_run, clfs, grid, prediction_windows):

    rv_lst = []

    time_periods = get_time_periods(df, prediction_windows)

    for period in time_periods:

        train_start_date = period[0]
        train_end_date = period[1]
        test_start_date = period[2]
        test_end_date = period[3]

        x_train, y_train, x_test, y_test = pipeline.get_train_test_splits(df, train_start_date, train_start_date, test_start_date, test_end_date)

        rv_row = clf_loop(models_to_run, clfs, grid, x_train, x_test, y_train, y_test, train_start_date, train_end_date, test_start_date, test_end_date)

        rv_lst.extend(rv_row)


    rv_df = pd.DataFrame(rv_lst, columns=('train_start','train_end','test_start','test_end','model_type','clf', 'parameters','baseline', 'auc-roc',
                                            'p_at_1', 'p_at_2', 'p_at_5', 'p_at_10', 'p_at_20', 'p_at_30', 'p_at_50', 'r_at_1', 'r_at_2','r_at_5',
                                            'r_at_10', 'r_at_20','r_at_30','r_at_50'))

    return rv_df



def clf_loop(models_to_run, clfs, grid, x_train, x_test, y_train, y_test, train_start_date, train_end_date, test_start_date, test_end_date):
    """Runs the loop using models_to_run, clfs, gridm and the data
    """
    rv = []

    for n in range(1, 2):
        # create training and valdation sets
        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            print(models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    fit = clf.fit(x_train, y_train.values.ravel())
                    y_pred_probs = fit.predict_proba(x_test)[:,1]
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test.values.ravel()), reverse=True))
                    row = [train_start_date, train_end_date, test_start_date, test_end_date,
                                models_to_run[index],clf, p,
                                baseline(y_test),
                               roc_auc_score(y_test, y_pred_probs),
                               precision_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                               precision_at_k(y_test_sorted,y_pred_probs_sorted,2.0),
                               precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                               precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                               precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                               precision_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                               precision_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                               recall_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                               recall_at_k(y_test_sorted,y_pred_probs_sorted,2.0),
                               recall_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                               recall_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                               recall_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                               recall_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                               recall_at_k(y_test_sorted,y_pred_probs_sorted,50.0)]
                               
                    rv.append(row)
                    # plot_precision_recall_n(y_test,y_pred_probs,clf)
                except IndexError as e:
                    print('Error:',e)
                    continue

    return rv


# a set of helper function to do machine learning evalaution

def joint_sort_descending(l1, l2):
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary

def precision_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    precision = precision_score(y_true, preds_at_k)
    return precision


def recall_at_k(y_true, y_scores, k):
    '''
    Calculates recall at given threshold k.
    '''
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    recall_at_k = generate_binary_at_k(y_scores, k)
    recall = recall_score(y_true, recall_at_k)
    return recall


def accuracy_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    #print(len(preds_at_k))

    pred = accuracy_score(y_true, preds_at_k)

    return pred


def f1_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)

    f1 = f1_score(y_true, preds_at_k)

    return f1

def baseline(y_test):
    base = y_test.sum()/ len(y_test)

    return base



def get_time_periods(df, prediction_windows):


    train_start_date = df['date_posted'].min()
    end_date = df['date_posted'].max()

    time_periods = []

    for window in prediction_windows:

        train_end_date = train_start_date + relativedelta(months=+window) - relativedelta(days=+1)

        while train_end_date + relativedelta(months=+window)<=end_date:

            test_start_date = train_end_date + relativedelta(days=+1)
            test_end_date = test_start_date + relativedelta(months=+window) - relativedelta(days=+1)
            
            # Build training and testing sets
            time_periods.append([train_start_date, train_end_date, test_start_date, test_end_date])

            # Increment time
            train_end_date += relativedelta(months=+window)


    return time_periods

