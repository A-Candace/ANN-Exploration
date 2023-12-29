# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 10:30:37 2023

@author: cagon
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime

# Setting the path to the folders containing the cluster data
# The predictors folder is comprised of spreadsheets. Each spreadsheet is named after a cluster (zone)
# In each spreadsheet, there is a Date column and a column for each predictor. 
# For each day, the dynamic predictor values are input.

predictors_folder = 'PredictorsClusterTopoX6'
response_folder = 'ResponseClusterTopoX6'
results_file = 'PoissonGLM_Results.csv'

# Defining start and end dates for training and testing
train_start_date = '2010-01-01 00:00:00'
train_end_date = '2018-01-01 00:00:00'
test_start_date = '2018-01-02 00:00:00'
test_end_date = '2019-12-31 00:00:00'

# Converting date strings to datetime objects
train_start_date = datetime.strptime(train_start_date, '%Y-%m-%d %H:%M:%S')
train_end_date = datetime.strptime(train_end_date, '%Y-%m-%d %H:%M:%S')
test_start_date = datetime.strptime(test_start_date, '%Y-%m-%d %H:%M:%S')
test_end_date = datetime.strptime(test_end_date, '%Y-%m-%d %H:%M:%S')

# Creating an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['Cluster', 'R2 Score'])

# Iterating over each cluster
for cluster_file in os.listdir(predictors_folder):
    cluster_id = os.path.splitext(cluster_file)[0]

    # Loading predictor data for the current cluster
    predictor_data = pd.read_excel(os.path.join(predictors_folder, cluster_file))

    # Filtering predictor data based on training and testing dates
    train_data = predictor_data[(predictor_data['Date'] >= train_start_date) & (predictor_data['Date'] <= train_end_date)]
    test_data = predictor_data[(predictor_data['Date'] >= test_start_date) & (predictor_data['Date'] <= test_end_date)]

    X_train = train_data[['SB', 'CB', 'PRCP', 'SNOW', 'MaxValue_Hourly']]
    X_test = test_data[['SB', 'CB', 'PRCP', 'SNOW', 'MaxValue_Hourly']]

    # Loading response data for the current cluster
    response_file = os.path.join(response_folder, f'{cluster_id}.xlsx')
    if os.path.exists(response_file):
        response_data = pd.read_excel(response_file)

        # Filtering response data based on training and testing dates
        y_train = response_data[(response_data['Date'] >= train_start_date) & (response_data['Date'] <= train_end_date)]['Flooding']
        y_test = response_data[(response_data['Date'] >= test_start_date) & (response_data['Date'] <= test_end_date)]['Flooding']

        # Normalizing predictor data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Creating and fitting the Poisson GLM model
        X_train = sm.add_constant(X_train)
        model = sm.GLM(y_train, X_train, family=sm.families.Poisson())
        results = model.fit()

        # Predicting on the testing set
        X_test = sm.add_constant(X_test)
        y_pred = results.predict(X_test)

        # Calculating R-squared for the testing set
        r2 = r2_score(y_test, y_pred)

        # Appending results to the DataFrame
        results_df = results_df.append({'Cluster': cluster_id, 'R2 Score': r2}, ignore_index=True)

        # Plotting the observed and predicted values
        plt.figure(figsize=(10, 6))
        plt.plot(y_test.values, label='Observed Street Counts')
        plt.plot(y_pred, label='Predicted Street Counts')
        plt.xlabel('Days')
        plt.ylabel('Counts')
        plt.title(f'Cluster {cluster_id}')
        plt.legend()
        plt.savefig(f'Cluster_{cluster_id}_plot.jpg')
        plt.close()
    else:
        print(f"Response file not found for cluster {cluster_id}")

# Saving results to a spreadsheet
results_df.to_csv(results_file, index=False)
