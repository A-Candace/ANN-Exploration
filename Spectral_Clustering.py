# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 12:53:03 2023

@author: cagon
"""

import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler

# Loaded the data from the Excel file
data = pd.read_excel('ClusteredDataCB_Spectral.xlsx')

# Selected the columns to be used for clustering
columns = ['Elevation', 'Slope', 'Imperv', 'BLDperArea', 'FTPperArea']

# Extracted the selected columns for clustering
X = data[columns]

# Standardized the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Performed Spectral Clustering
n_clusters = 6  # Number of clusters
clustering = SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors')
labels = clustering.fit_predict(X_scaled)

# Added the cluster labels to the data
data['Cluster'] = labels

# Saved the updated data to a new CSV file
data.to_csv('ClusteredDataCB_Spectral.csv', index=False)
