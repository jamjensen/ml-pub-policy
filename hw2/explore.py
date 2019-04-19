import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import pipeline

COL_NAMES = ['NumberOfTimes90DaysLate', 'DebtRatio', 'age']

LOC = 'zipcode'
TARGET = 'SeriousDlqin2yrs'

FILENAME = 'credit-data.csv'



def go():

	df = pipeline.load_data(FILENAME)

	summarize(df, COL_NAMES)


## Distributions of Different Variables

def summarize(df, cols):

	for col in cols:
		print(df[col].describe())


def target_loc_dist(target, loc):
	'''
	Percentage of people who have had deliquency
	'''

	one = df[df[target] == 1].groupby(loc).size()
	two = df.groupby(target).PersonID.nunique()

	one / two

## Make heatmap

def make_heatmap(df):

	corr = df.corr()
	ax = sns.heatmap(corr, xticklabels=corr.columns.values, 
                yticklabels=corr.columns.values)

	return ax


## Visualize missing data

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





