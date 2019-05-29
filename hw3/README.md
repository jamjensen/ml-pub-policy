Assignment #3 - Improving the Pipeline
---

## Goal 

The problem is to predict if a project on donorschoose will not get fully funded within 60 days of posting. The data is a file that has one row for each project posted with a column for "date_posted" (the date the project was posted) and a column for "datefullyfunded" (the date the project was fully funded. The task is improve upon on our previous assignments, building out the machine learning pipeline to predict if a project on donorschoose will not get fully funded within 60 days of posting.

## Data

The dataset used for this exercise is a modified version of the [Kaggle DonorsChoose Dataset](https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose/data) and can be found in data/projects_2012_2013.csv.

## Usage

The Jupyter Notebook, titled hw3-notebook, walks through the process of loading the data, exploring the variables, creating the features, selecting the models to run, and finally runnning the models. The notebook calls pipeline.py, which handles data loading, processing, feature generation, and creation of test/train datasets. It also calls loops.py, which takes the length of each time period, and passes training and testing datasets through a given list of models. The run_time_loop() function returns a table with results across train test splits over time and performance metrics (baseline, precision and recall at different thresholds 1%, 2%, 5%, 10%, 20%, 30%, 50% and AUC_ROC). 
