#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
data_matrix = pd.read_csv("FINAL Data MSD.csv")
data_matrix = data_matrix.drop(columns=['Ground','Date','Match_ID','Opposition'])
df=data_matrix.dropna(axis=1)

print(data_matrix)
data=data_matrix


# In[2]:


#Q2)Load the dataset, drop all the null records and replace the NA values in the numerical column with the mean value of the field as per the class label and categorical columns with the mode value of the field as per the class label.
import pandas as pd
#Helper Function
def count_nan(dataframe):
  nan_count = dataframe.isna().sum()
  nan_info = pd.DataFrame({'Column Name': dataframe.columns, 'NaN Count': nan_count})
  return nan_count
# Load the data
data_matrix = pd.read_csv("FINAL Data MSD.csv", header='infer')
df = data_matrix.drop(columns=['Ground','Date','Match_ID','Opposition','Dismissal'])
df['Position'] = df['Position'].replace('DNB', np.nan)
df['Inns'] = df['Inns'].replace('DNB', np.nan)
df['Runs'] = df['Runs'].replace('DNB', np.nan)
print("Nan_CountBeforePre-processing:",count_nan(df))
#Pre processing
df2=df.copy()
df2=df2.dropna(axis=0)
Mean_Cols=['Runs','Inns', 'Position']

Mode_Cols=['Dismissal','Match_Type']

for column in df.columns:
    if column in Mode_Cols:
        mode_value = df[column].mode()[0]  # Calculate mode
        df[column].fillna(mode_value, inplace=True)
    elif column in Mean_Cols:
            mean_value = df[column].astype(float).mean()  # Calculate mean
            df[column].fillna(mean_value, inplace=True)
print("Nan_CountAfterPre-processing:",count_nan(df))


# In[3]:


#Q3) Perform statistical analysis on the selected dataset (count, sum, range, min, max, mean, median, mode, variance and Standard deviation).
def count(cols):
  print("Counts:")
  for col in cols:
    output=df[col].value_counts()
    print(output)
    print()
def sum_min_max_range(df, cols):
    print("Sum, Min, Max, and Range:")
    for col in cols:
        col_data = df[col].astype(float)
        col_sum = col_data.sum()
        col_min = col_data.min()
        col_max = col_data.max()
        col_range = col_max - col_min
        print(f"{col}: Sum={col_sum}, Min={col_min}, Max={col_max}, Range={col_range}")
    print()

def mode(df, cols):
    print("Mode:")
    for col in cols:
        col_mode = df[col].mode()
        print(f"{col}: {col_mode.values}")
    print()

def variance_stddev(df, cols):
    print("Variance and Standard Deviation:")
    for col in cols:
        # Check if column contains numeric data
        if df[col].dtype in ['int64', 'float64']:
            # Drop missing values before calculating variance and standard deviation
            col_data = df[col].dropna().astype(float)
            col_var = col_data.var()
            col_std = col_data.std()
            print(f"{col}: Variance={col_var}, Standard Deviation={col_std}")
        else:
            print(f"{col}: Contains non-numeric data")
        print()


count_cols=['Match_Type']
mean_median_cols=['Runs','BF','4s','6s','SR']
mode_cols=['Match_Type']
sum_min_max_range_cols=['Runs','BF','4s','6s','SR']
variance_stddev_cols=['Runs','BF','4s','6s','SR']
count(count_cols)
#mean_median(mean_median_cols)
mode(df,mode_cols)
sum_min_max_range(df,sum_min_max_range_cols)
variance_stddev(df,variance_stddev_cols)


# In[4]:


#Q3 Display all the unique value counts and unique values of all the columns of the dataset.

for column in df.columns:
   unique_values = df[column].unique()
   value_counts = df[column].value_counts()
   print(f"Column: {column}")
   print(f"Unique Values: {unique_values}")
   print(f"Value Counts:\n{value_counts}\n")


# In[5]:


#Q4 Draw applicable plots to visualise data using the subplot concept on the dataset. (scatter plot/ line graph/ histogram etc.)
import matplotlib.pyplot as plt

# Convert 'Match_Type' column to string
df['Match_Type'] = df['Match_Type'].astype(str)

numeric_columns = ['Runs', 'BF', '4s', '6s', 'SR']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Define colors for each match type
color_map = {'ODI': 'r', 'Test': 'g', 'T20': 'b'}
color_list = df['Match_Type'].map(color_map)

# Scatterplots:
plot_idx = 1
plt.subplots_adjust(hspace=3, wspace=2)
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20, 16))
for x in range(len(numeric_columns)):
    for y in range(x + 1, len(numeric_columns)):
        plt.subplot(5, 2, plot_idx)
        plt.xlabel(numeric_columns[x])
        plt.ylabel(numeric_columns[y])
        plt.scatter(df[numeric_columns[x]], df[numeric_columns[y]], s=4, c=color_list, alpha=0.85)
        plot_idx += 1

plt.show()


# In[6]:


#Q5 Train the model of the K-nearest Neighbors Classifier/Regressor with 80% of the data and predict the class label for the rest 20% of the data. Evaluate the model with all appropriate measures.
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


X = df.iloc[:, :-1]
Y= df.iloc[:, -1]
# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, stratify=Y)
best_accuracy = 0
best_k = 0
for k in range(1, 51):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k
print("best_k:",best_k)
print("best_accuracy:",best_accuracy*100)

