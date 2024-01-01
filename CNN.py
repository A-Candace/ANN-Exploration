# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 21:45:57 2023

@author: cagon
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


timeseries3cluster_folder = 'PredictorsClusterTopoX6'
timeseries4cluster_folder = 'ResponseClusterTopoX6'


results_df = pd.DataFrame(columns=['Cluster', 'R2 Score', 'RMSE'])

# Definuing the CNN model
class CNNModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=1)  # Adjust kernel size
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1)  # Adjust kernel size
        self.fc = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Transpose to match Conv1d input format
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.mean(dim=2)  # Global average pooling over time dimension
        x = self.fc(x)
        return x



train_start_date = '2010-01-01 00:00:00'
train_end_date = '2018-01-01 00:00:00'
test_start_date = '2018-01-02 00:00:00'
test_end_date = '2019-12-31 00:00:00'


for cluster_file in os.listdir(timeseries3cluster_folder):
    cluster_id = os.path.splitext(cluster_file)[0]

    
    predictor_data = pd.read_excel(os.path.join(timeseries3cluster_folder, cluster_file))
    
    
    predictor_data['Date'] = pd.to_datetime(predictor_data['Date'], format='%Y-%m-%d %H:%M:%S')
    train_predictor_data = predictor_data[(predictor_data['Date'] >= train_start_date) & (predictor_data['Date'] <= train_end_date)]
    test_predictor_data = predictor_data[(predictor_data['Date'] >= test_start_date) & (predictor_data['Date'] <= test_end_date)]
    
    X_train = train_predictor_data[['SB', 'CB', 'PRCP', 'SNOW', 'MaxValue_Hourly']].values
    X_test = test_predictor_data[['SB', 'CB', 'PRCP', 'SNOW', 'MaxValue_Hourly']].values

    
    target_file = os.path.join(timeseries4cluster_folder, f'{cluster_id}.xlsx')
    target_data = pd.read_excel(target_file)
    
    
    target_data['Date'] = pd.to_datetime(target_data['Date'], format='%Y-%m-%d %H:%M:%S')
    train_target_data = target_data[(target_data['Date'] >= train_start_date) & (target_data['Date'] <= train_end_date)]
    test_target_data = target_data[(target_data['Date'] >= test_start_date) & (target_data['Date'] <= test_end_date)]
    
    y_train = train_target_data['Flooding'].values
    y_test = test_target_data['Flooding'].values

    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)

    
    input_size = X_train.shape[1]
    output_size = 1
    model = CNNModel(input_size, output_size)

    
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    
    num_epochs = 1000
    batch_size = 32
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train.unsqueeze(1))  # Add a channel dimension
        loss = criterion(outputs, y_train.unsqueeze(1))
        loss.backward()
        optimizer.step()

    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test.unsqueeze(1))  # Add a channel dimension
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    
    results_df = results_df.append({
        'Cluster': cluster_id,
        'R2 Score': r2,
        'RMSE': rmse
    }, ignore_index=True)
    
    plt.plot(y_test, label='Actual Street Counts')
    plt.plot(y_pred.numpy().flatten(), label='Predicted Street Counts')
    plt.xlabel('Days')
    plt.ylabel('Counts')
    plt.title(f'Cluster {cluster_id}')
    plt.legend()
    plt.savefig(f'Cluster_{cluster_id}_plot.jpg')
    plt.close()
    
    print(f"Cluster: {cluster_id}")
    print("Predicted Street Counts:", y_pred.numpy().flatten())
    print("Testing Street Counts:", y_test.numpy())


results_df.to_csv('CNN.csv', index=False)
