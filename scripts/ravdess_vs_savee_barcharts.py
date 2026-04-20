#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

# Build the data table
data = {
    'model': ['XGBoost', 'LogisticRegression', 'RandomForest', 'EllipticEnvelop', 'LocalOutlierFactor', 'OneClassSVM', 'IsolationForest'],
    'ravdess': [0.42, 0.38, 0.31, 0.31, 0.24, 0.23, 0.21],
    'savee': [0.79, 0.84, 0.36, 0.56, 0.77, 0.73, 0.63]
}


df = pd.DataFrame(data)
# df.sort_values(by='savee', ascending=False)

# Build the plot
ax = df.plot(x='model', kind='bar', figsize=(10, 6), rot=45)

plt.title('Model accuracy comparison on RAVDESS and SAVEE')
plt.ylabel('Score')
plt.xlabel('Model')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()

data = {
    'dataset': ['SAVEE', 'RAVDESS'],
    # 'total_samples': [480, 1440],
    'anomalies': [360, 1260],
    'normal': [120, 180]
}


df = pd.DataFrame(data)
# df.sort_values(by='savee', ascending=False)

# Build the plot
ax = df.plot(x='dataset', kind='bar', figsize=(5, 6), rot=45)

plt.title('Class distribution in RAVDESS and SAVEE')
plt.ylabel('Number of samples')
plt.xlabel('Dataset')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()
