import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime

# Setting the path to the folders containing the cluster data
timeseries3cluster_folder = 'PredictorsClusterTopoX6'
timeseries4cluster_folder = 'ResponseClusterTopoX6'

# Creating an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['Cluster', 'R2 Score', 'RMSE'])

# Defininng the Recurrent Neural Network (RNN) model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, sequence_length=6):
        super(RNN, self).__init__()
        self.sequence_length = sequence_length
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Select the last time step's output
        return out

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

# Iterating over each cluster in timeseries3cluster folder
for cluster_file in os.listdir(timeseries3cluster_folder):
    cluster_id = os.path.splitext(cluster_file)[0]

    # Loading the predictor data for the current cluster from timeseries3cluster
    predictor_data = pd.read_excel(os.path.join(timeseries3cluster_folder, cluster_file))
    
    # Filtering predictor data based on training and testing dates
    predictor_train_data = predictor_data[(predictor_data['Date'] >= train_start_date) & (predictor_data['Date'] <= train_end_date)]
    predictor_test_data = predictor_data[(predictor_data['Date'] >= test_start_date) & (predictor_data['Date'] <= test_end_date)]
    
    # Loading the target data for the current cluster from timeseries4cluster
    target_file = os.path.join(timeseries4cluster_folder, f'{cluster_id}.xlsx')
    target_data = pd.read_excel(target_file)
    y = target_data['Flooding'].values

    # Splitting the data into training and testing sets
    X_train = predictor_train_data[['SB','CB','PRCP','SNOW','MaxValue_Hourly']].values
    y_train = y[(predictor_data['Date'] >= train_start_date) & (predictor_data['Date'] <= train_end_date)]
    X_test = predictor_test_data[['SB','CB','PRCP','SNOW','MaxValue_Hourly']].values
    y_test = y[(predictor_data['Date'] >= test_start_date) & (predictor_data['Date'] <= test_end_date)]

    # Normalizing the predictor data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Converting the data to PyTorch tensors
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)

    # Defining the RNN model
    input_size = X_train.shape[1]
    #hidden_size = 64
    hidden_size = 16
    #hidden_size = 32
    output_size = 1
    #model = RNN(input_size, hidden_size, output_size)
    model = RNN(input_size, hidden_size, output_size, sequence_length=6)
    # Define the loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.007)

    # Trainimg the model
    num_epochs = 1000
    batch_size = 32
    #batch_size = 32
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train.unsqueeze(1))  # Add a time step dimension
        loss = criterion(outputs, y_train.unsqueeze(1))
        loss.backward()
        optimizer.step()

    # Evaluating the model on the testing set
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test.unsqueeze(1))
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Appending the results to the DataFrame
    results_df = results_df.append({
        'Cluster': cluster_id,
        'R2 Score': r2,
        'RMSE': rmse
    }, ignore_index=True)
    # Plotting the actual and predicted values as line plots
    plt.plot(y_test, label='Actual Street Counts')
    plt.plot(y_pred.numpy().flatten(), label='Predicted Street Counts')
    plt.xlabel('Days')
    plt.ylabel('Counts')
    plt.title(f'Cluster {cluster_id}')
    plt.legend()
    plt.savefig(f'Cluster_{cluster_id}_plot.jpg')
    plt.close()
    # Printing the predicted and testing street counts
    print(f"Cluster: {cluster_id}")
    print("Predicted Street Counts:", y_pred.numpy().flatten())
    print("Testing Street Counts:", y_test.numpy())

# Saving the results to a CSV file with a single sheet
results_df.to_csv('RNN.csv', index=False)
