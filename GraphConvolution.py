# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 17:48:05 2024

@author: cagon
"""


# Importing necessary libraries
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

# Define the folder paths and other parameters
predictor_folder = 'Predictors'
target_folder = 'Response'
cluster_info_path = 'ClusterInformation.xlsx'
train_start_date = '2010-01-01 00:00:00'
train_end_date = '2018-01-01 00:00:00'
test_start_date = '2018-01-02 00:00:00'
test_end_date = '2019-12-31 00:00:00'
num_predictors = 5  # Number of predictor variables

# Load cluster information for node positions
cluster_info = pd.read_excel(cluster_info_path)

# Define a Graph Neural Network (GNN
# Number of nodes (based on the number of files in the folders)
num_nodes = 6

# Define the Graph Neural Network (GNN) model
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Define a function to load and preprocess data for a given node
def load_and_preprocess_data(node, train_start, train_end, test_start, test_end):
    # Load predictor data
    predictor_path = os.path.join(predictor_folder, f'{node}.xlsx')
    predictors = pd.read_excel(predictor_path, parse_dates=['Date'])

    # Load response data
    response_path = os.path.join(target_folder, f'{node}.xlsx')
    response = pd.read_excel(response_path, parse_dates=['Date'])

    # Merge predictor and response data
    data = pd.merge(predictors, response, on='Date')

    # Filter data based on date ranges
    train_data = data[(data['Date'] >= train_start) & (data['Date'] <= train_end)]
    test_data = data[(data['Date'] >= test_start) & (data['Date'] <= test_end)]

    # Separate features (predictors) and target (Flooding)
    X_train = train_data.iloc[:, 1:num_predictors + 1].values
    y_train = train_data['Flooding'].values
    X_test = test_data.iloc[:, 1:num_predictors + 1].values
    y_test = test_data['Flooding'].values

    return X_train, y_train, X_test, y_test

# Create a fully connected graph (all nodes connected to all other nodes)
edge_index = torch.tensor([(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j], dtype=torch.long).t()

# Initialize a list to store R^2 scores for each node
r2_scores = []

# Loop through each node
for node in range(num_nodes):
    # Load and preprocess data for the current node
    X_train, y_train, X_test, y_test = load_and_preprocess_data(node, train_start_date, train_end_date, test_start_date, test_end_date)

    # Create a Data object for the GNN
    data = Data(x=torch.tensor(X_train, dtype=torch.float32), edge_index=edge_index, y=torch.tensor(y_train, dtype=torch.float32))

    # Initialize the GNN model
    input_dim = num_predictors
    hidden_dim = 64
    output_dim = 1
    model = GNNModel(input_dim, hidden_dim, output_dim)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()

    # Testing
    model.eval()
    test_data = Data(x=torch.tensor(X_test, dtype=torch.float32), edge_index=edge_index, y=torch.tensor(y_test, dtype=torch.float32))
    with torch.no_grad():
        y_pred = model(test_data)

    # Calculate R^2 score for the current node
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)

# Print the R^2 scores for each node
for node, r2 in enumerate(r2_scores):
    print(f"Node {node}: R^2 Score = {r2}")