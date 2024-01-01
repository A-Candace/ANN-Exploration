# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 13:05:11 2023

@author: cagon
"""


import os
import pandas as pd
import numpy as np
import networkx as nx
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Load ClusterInformation.csv to get latitude and longitude
cluster_info = pd.read_excel('ClusterInformation.xlsx')

# Define your graph based on latitude and longitude
def create_graph_from_coordinates(cluster_info):
    G = nx.Graph()
    for idx, row in cluster_info.iterrows():
        node_name = row['Cluster']
        latitude = row['lat']
        longitude = row['lng']
        G.add_node(node_name, latitude=latitude, longitude=longitude)
    
    # Define edges based on geographical proximity or any other criteria
    # For example, you can connect nodes if they are within a certain distance.
    # Here, we connect nodes if they are within 0.1 degrees of latitude or longitude.
    for u, v, data in G.edges(data=True):
        u_lat, u_lng = G.nodes[u]['latitude'], G.nodes[u]['longitude']
        v_lat, v_lng = G.nodes[v]['latitude'], G.nodes[v]['longitude']
        if abs(u_lat - v_lat) <= 0.005 or abs(u_lng - v_lng) <= 0.005:
        #if abs(u_lat - v_lat) <= 0.5 or abs(u_lng - v_lng) <= 0.5:
            G[u][v]['weight'] = 1.0  # You can assign a weight to the edges

    return G

# Load predictor and target data
def load_data(predictor_folder, target_folder):
    predictor_data = {}
    target_data = {}
    for node_name in os.listdir(predictor_folder):
        predictor_file = os.path.join(predictor_folder, node_name)
        if os.path.isfile(predictor_file) and predictor_file.endswith('.xlsx'):
            node_df = pd.read_excel(predictor_file)
            predictor_data[node_name] = node_df
            
    for node_name in os.listdir(target_folder):
        target_file = os.path.join(target_folder, node_name)
        if os.path.isfile(target_file) and target_file.endswith('.xlsx'):
            node_df = pd.read_excel(target_file)
            target_data[node_name] = node_df
    
    return predictor_data, target_data

# Split data into training and test sets based on date ranges
def split_data(predictor_data, target_data, start_date, end_date):
    train_predictors = {}
    test_predictors = {}
    train_targets = {}
    test_targets = {}
    
    for node_name, predictor_df in predictor_data.items():
        target_df = target_data.get(node_name)
        if target_df is not None:
            mask_train = (predictor_df['Date'] >= start_date) & (predictor_df['Date'] < end_date)
            mask_test = (predictor_df['Date'] >= end_date)
            
            train_predictors[node_name] = predictor_df[mask_train]
            test_predictors[node_name] = predictor_df[mask_test]
            
            train_targets[node_name] = target_df[mask_train]
            test_targets[node_name] = target_df[mask_test]
    
    return train_predictors, test_predictors, train_targets, test_targets

# Build the adjacency matrix (graph) based on coordinates
G = create_graph_from_coordinates(cluster_info)

# Load data and split into training and test sets
predictor_folder = 'PredictorsClusterTopoX6'
target_folder = 'ResponseClusterTopoX6'
start_date = '2010-01-01'
end_date = '2018-01-02'  # Adjust as needed

predictor_data, target_data = load_data(predictor_folder, target_folder)
train_predictors, test_predictors, train_targets, test_targets = split_data(
    predictor_data, target_data, start_date, end_date
)

# Preprocess your predictor and target data as needed

# Define and build your GNN model using TensorFlow and Keras
def build_gnn_model(input_shape, num_output_nodes):
    model = keras.Sequential([
        # Define your GNN layers here
        # Example: layers.GCNConv(32, activation='relu'),
        # Add more layers as needed

        
        ###keras.layers.GATConv(32, activation='relu')

        keras.layers.Dense(32, activation='relu'),  # Adjust parameters as needed
        #keras.layers.Dense(64, activation='relu'),  # Add more layers as needed
        keras.layers.Dense(32, activation='relu'), 
        #keras.layers.Dense(32, activation='relu'), 
        # Output layer
        keras.layers.Dense(num_output_nodes)  # Adjust the number of output nodes

        # Output layer
        #keras.layers.GraphConv(32, activation='relu')

        #keras.layers.Dense(num_output_nodes)  # Adjust the number of output nodes
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')  # Adjust the loss function as needed
    return model

# Define GNN model hyperparameters
predictor_dim = 5  # Replace with the actual number of predictor features
input_shape = (None, predictor_dim)  # Adjust the input shape based on your data
num_output_nodes = 1  # Adjust for your regression task
input_shape = (None, predictor_dim)  # Adjust the input shape based on your data
num_output_nodes = 1  # Adjust for your regression task

# Build the GNN model
gnn_model = build_gnn_model(input_shape, num_output_nodes)

from sklearn.metrics import r2_score

# Train the GNN model
for node_name, train_predictor_df in train_predictors.items():
    train_target_df = train_targets[node_name]
    train_predictors_array = train_predictor_df.drop('Date', axis=1).values
    train_targets_array = train_target_df['Flooding'].values
    
    # Train the model for each node
    gnn_model.fit(train_predictors_array, train_targets_array, epochs=50)  # Adjust epochs and batch size

# Evaluate the GNN model on the test set
for node_name, test_predictor_df in test_predictors.items():
    test_target_df = test_targets[node_name]
    test_predictors_array = test_predictor_df.drop('Date', axis=1).values
    test_targets_array = test_target_df['Flooding'].values
    
    # Evaluate the model for each node
    mse = mean_squared_error(test_targets_array, gnn_model.predict(test_predictors_array))
    print(f"Node {node_name} - Mean Squared Error: {mse}")
    
for node_name, test_predictor_df in test_predictors.items():
    test_target_df = test_targets[node_name]
    test_predictors_array = test_predictor_df.drop('Date', axis=1).values
    test_targets_array = test_target_df['Flooding'].values
    
    # Evaluate the model for each node
    mse = mean_squared_error(test_targets_array, gnn_model.predict(test_predictors_array))
    r2 = r2_score(test_targets_array, gnn_model.predict(test_predictors_array))  # Calculate R2 score
    print(f"Node {node_name} - Mean Squared Error: {mse}, R2 Score: {r2}")
# You can also use the trained model to make predictions for new data

# Visualize results on a map (optional)
# Use latitude and longitude information from cluster_info and predicted values
# You can use libraries like matplotlib or folium for visualization
