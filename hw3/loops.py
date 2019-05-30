
from __future__ import division
import pandas as pd
import numpy as np
from sklearn import preprocessing, svm, metrics, tree, decomposition, svm
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
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


EMPTY_DF = pd.DataFrame(columns=('train_start','train_end','test_start','test_end','model_type','clf', 'parameters','baseline', 'auc-roc',
                                            'f1_at_5','f1_at_10','f1_at_20','f1_at_30','f1_at_50','a_at_5', 'a_at_10','a_at_20','a_at_30','a_at_50',
                                            'p_at_1', 'p_at_2', 'p_at_5', 'p_at_10', 'p_at_20', 'p_at_30', 'p_at_50', 'r_at_1', 'r_at_2','r_at_5',
                                            'r_at_10', 'r_at_20','r_at_30','r_at_50'))

# following functions are heavily influenced by code found here: https://github.com/rayidghani/magicloops/blob/master/magicloop.py and /simpleloop.py
def define_clfs_params(grid_size):
    """Define defaults for different classifiers.
    Define three types of grids:
    Test: for testing your code
    Small: small grid
    Large: Larger grid that has a lot more parameter sweeps
    """

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': LinearSVC(random_state=0),
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
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
    'BG': {'n_estimators' : [10, 20], 'max_samples' : [.25, .5]}
           }

    
    small_grid = { 
    'RF':{'n_estimators': [1,10,100,1000], 'max_depth': [5,10], 'max_features': ['sqrt','log2'],'min_samples_split': [5,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.01, .1, 1]},
    'GB': {'n_estimators': [1,10,100], 'learning_rate' : [.01, 0.1], 'subsample' : [0.1, 0.5], 'max_depth': [5, 10]},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [5, 10, 20],'min_samples_split': [5,10]},
    'SVM' :{'C' :[0.01]},
    'KNN' :{'n_neighbors': [2,5,10],'weights': ['uniform'],'algorithm': ['auto']},
    'BG': {'n_estimators' : [2, 10, 20], 'max_samples' : [.1, .5]}
           }

    if (grid_size == 'large'):
        return clfs, large_grid
    elif (grid_size == 'small'):
        return clfs, small_grid
    else:
        return None



def run_time_loop(df, models_to_run, clfs, grid, prediction_windows, output_type):

    results_df = EMPTY_DF

    time_periods = get_time_periods(df, prediction_windows)

    for period in time_periods:

        train_start_date = period[0]
        train_end_date = period[1]
        test_start_date = period[2]
        test_end_date = period[3]

        x_train, y_train, x_test, y_test = pipeline.get_train_test_splits(df, train_start_date, train_end_date, test_start_date, test_end_date)

        output_df = clf_loop(models_to_run, clfs, grid, x_train, x_test, y_train, y_test, train_start_date, train_end_date, test_start_date, test_end_date, output_type)

        results_df = results_df.append(output_df, ignore_index=True)


    return results_df


def clf_loop(models_to_run, clfs, grid, x_train, x_test, y_train, y_test, train_start_date, train_end_date, test_start_date, test_end_date, output_type):
    """Runs the loop using models_to_run, clfs, gridm and the data
    """
    inner_df = EMPTY_DF.copy()

    for n in range(1, 2):
        # create training and valdation sets
        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            print(models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    fit = clf.fit(x_train, y_train.values.ravel())
                    if models_to_run[index] == 'SVM':
                        y_pred_probs = fit.decision_function(x_test)
                    else:
                        y_pred_probs = fit.predict_proba(x_test)[:,1]
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test.values.ravel()), reverse=True))
                    inner_df.loc[len(inner_df)] = [train_start_date, train_end_date, test_start_date, test_end_date,
                                models_to_run[index],clf, p,
                                baseline(y_test),
                               roc_auc_score(y_test, y_pred_probs),
                               f1_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                               f1_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                               f1_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                               f1_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                               f1_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                               accuracy_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                accuracy_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                accuracy_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                accuracy_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                                accuracy_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
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
                               
                    plot_precision_recall_n(y_test,y_pred_probs,clf, output_type, p, train_end_date)
                except IndexError as e:
                    print('Error:',e)
                    continue

    return inner_df



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

def plot_precision_recall_n(y_true, y_prob, model, output_type, p, train_end_date):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax2.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = str(model).split('(')[0] + str(p) + str(train_end_date)
    plt.title(name)

    if (output_type == 'save'):
        plt.savefig('Plots2/' + name +'.png')
    elif (output_type == 'show'):
        plt.show()
    else:
        plt.show()

# Inspiration for get_time_periods()
# https://github.com/rayidghani/magicloops/blob/master/temporal_validate.py
def get_time_periods(df, prediction_windows):
    '''
    Given a dataframe with a relevant date column and a list of integers, representing
    the window in which testing datasets should occur, this function returns
    a list of time periods. Within each time period is the start and end date for the
    testing and datasets. The output of this function is used to create the training
    and testing datasets. In order to predict whether or not a project will be funded in
    60 days, we need to have a gap of 60 days between end of training set and the start
    of the testing set. 
    '''
    # test_size =

    train_start_date = df['date_posted'].min()
    end_date = df['date_posted'].max()
    # end_date = end_date - relativedelta(days=+60)

    time_periods = []

    for window in prediction_windows:

        train_end_date = train_start_date + relativedelta(months=+window) - relativedelta(days=60)

        while train_end_date + relativedelta(months=+window)<=end_date:

            test_start_date = train_end_date + relativedelta(days=+60)
            test_end_date = test_start_date + relativedelta(months=+window) - relativedelta(days=+60)
            
            time_periods.append([train_start_date, train_end_date, test_start_date, test_end_date])
            train_end_date += relativedelta(months=+window) 


    return time_periods

