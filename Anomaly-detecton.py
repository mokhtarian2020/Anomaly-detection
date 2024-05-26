import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('transactions.csv')

# Data preprocessing
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['amount', 'transaction_time']])

# DBSCAN model
dbscan = DBSCAN(eps=0.5, min_samples=10)
dbscan_labels = dbscan.fit_predict(scaled_data)

# Z-Score method
data['z_score_amount'] = (data['amount'] - data['amount'].mean()) / data['amount'].std()
data['z_score_time'] = (data['transaction_time'] - data['transaction_time'].mean()) / data['transaction_time'].std()
data['z_score'] = np.sqrt(data['z_score_amount']**2 + data['z_score_time']**2)

# Identify anomalies
anomalies_dbscan = data[dbscan_labels == -1]
anomalies_zscore = data[data['z_score'] > 3]

# Plotting results
plt.figure(figsize=(12, 6))

# DBSCAN plot
plt.subplot(1, 2, 1)
plt.scatter(data['transaction_time'], data['amount'], c=dbscan_labels, cmap='Paired', s=5)
plt.title('DBSCAN Anomaly Detection')
plt.xlabel('Transaction Time')
plt.ylabel('Amount')

# Z-Score plot
plt.subplot(1, 2, 2)
plt.scatter(data['transaction_time'], data['amount'], c=(data['z_score'] > 3), cmap='Paired', s=5)
plt.title('Z-Score Anomaly Detection')
plt.xlabel('Transaction Time')
plt.ylabel('Amount')

plt.tight_layout()
plt.show()



