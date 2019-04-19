import pandas as pd
import seaborn as sns 


## Distributions of Different Variables



## Make heatmap

def make_heatmap(df):

	corr = df.corr()
	ax = sns.heatmap(corr, xticklabels=corr.columns.values, 
                yticklabels=corr.columns.values)

	return ax