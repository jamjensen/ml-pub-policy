import pandas as pd
import numpy as np
import seaborn as sns
# import joypy
# import missingno as msno
import pipeline
import matplotlib.pyplot as plt



COL_NAMES = ['NumberOfTimes90DaysLate', 'MonthlyIncome', 'age']
LOC = 'zipcode'
TARGET = 'SeriousDlqin2yrs'


def summarize(df, cols):
	'''
	Dispalys the moments and quartile ranges for a given
	list of varables
	Inputs:
		df: dataframe
		cols: list of columns
	'''

	for col in cols:
		print(df[col].describe())


def target_loc_dist(df, target, loc):
	'''
	Percentage of people who have had deliquency
	Inputs:
		target (str): name of target column
		loc (str): name of column with record location
	'''

	one = df[df[target] == 1].groupby(loc).size()
	two = df.groupby(loc).PersonID.nunique()

	return one / two


def make_histogram(df, col, kde = True, rug = False, bins = None):
	'''
	Takes the column (str) of a dataframe and plots distribution of its values.
	Continous variables. 
	'''

	# Note: NaN's are exluded
	plot_var = df[col]
	if bins is None:
		sns.distplot(plot_var[~plot_var.isnull()], kde = kde, rug = rug)
	else:
		sns.distplot(plot_var[~plot_var.isnull()], kde = kde, rug = rug, bins = bins)
	plt.title(col + ' Histogram')
	plt.ylabel('Frequency')
	plt.show()


def make_heatmap(df):
	'''
	Takes a dataframe and makes a heatmap visualizing correlations among variables
	'''

	corr = df.corr()
	ax = sns.heatmap(corr, xticklabels=corr.columns.values, 
                yticklabels=corr.columns.values)

	return ax


def missing_bar(df):
	'''
	Visualize total counts of missing values
	'''

	ax = msno.bar(df, figsize=[9.,7.])

	return ax


def missing_corr(df):
	'''
	Find correlations among missing data
	'''

	ax = msno.heatmap(df) 

	return ax

def missing_matrix(df):
	'''
	Visualize variation in occurence by index of missing values
	'''

	ax = msno.matrix(df, figsize=[9.,7.])

	return ax

def null_cols(df):

	isnull = df.isnull().any()
	null_cols = list(isnull[isnull == True].index)

	return null_cols

