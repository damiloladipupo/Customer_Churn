import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from pandas import read_csv
# from pandas.plotting import scatter_matrix
mydf = pd.read_csv("customer_churn.csv")
print(mydf)

# checking the summary of the dataset
print("\nSummary Statistics of Customer Churn Dataset:")
print(mydf.describe())

#checking for the first few rows of the dataset
print("\nFirst Few Rows of Customer Churn  Dataset:")
print(mydf.head())

# I Checked for null  values in the datset here
print("\nChecking for Missing Values in the Dataset:")
print(mydf.isnull().sum())

print("\nChecking the information of the Dataset:")
print(mydf.info())

