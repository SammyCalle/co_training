import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

with open("../YearExperimentResults/min_max_scaler/balanced/metrics.pkl", 'rb') as f:
    metrics_balanced_min_max = pickle.load(f)

with open("../YearExperimentResults/min_max_scaler/metrics.pkl", 'rb') as f:
    metrics_min_max = pickle.load(f)

with open("../YearExperimentResults/standard_scaler/metrics.pkl", 'rb') as f:
    metrics_standard_scaler = pickle.load(f)

with open("../YearExperimentResults/non_normal/metrics.pkl", 'rb') as f:
    metrics_non_normal = pickle.load(f)

accuracy_df = pd.concat([metrics_balanced_min_max['Accuracy'],
                         metrics_min_max['Accuracy'],
                         metrics_standard_scaler['Accuracy'],
                         metrics_non_normal['Accuracy']], axis=1)

accuracy_df.columns = ['Balanced Min-Max', 'Min-Max', 'Standard Scaler', 'Non-Normal']

mean_metrics_balanced_min_max = accuracy_df['Balanced Min-Max'].mean()
mean_metrics_min_max = accuracy_df['Min-Max'].mean()
mean_metrics_standard_scaler = accuracy_df['Standard Scaler'].mean()
mean_metrics_non_normal = accuracy_df['Non-Normal'].mean()

accuracy_df.plot(kind='line', figsize=(10, 6))
plt.legend(labels=[f'Balanced Min-Max (Avg: {mean_metrics_balanced_min_max:.2f})',
                   f'Min-Max (Avg: {mean_metrics_min_max:.2f})',
                   f'Standard Scaler (Avg: {mean_metrics_standard_scaler:.2f})',
                   f'Non-Normal (Avg: {mean_metrics_non_normal:.2f})'],
           loc='lower right')

plt.title('Accuracy Comparison')
plt.xlabel('Experiment')
plt.ylabel('Accuracy')
plt.show()