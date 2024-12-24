import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

with open("../Online/YearExperimentResults/metrics.pkl", 'rb') as f:
    online_non_normal_non_optimized = pickle.load(f)

with open("../NormalAndBatch/YearExperimentResults/Normal/metrics.pkl", 'rb') as f:
    normal_non_normal_non_optimized = pickle.load(f)

with open("../NormalAndBatch/YearExperimentResults/Batch/metrics.pkl", 'rb') as f:
    batch_non_normal_non_optimized = pickle.load(f)


accuracy_df = pd.concat([online_non_normal_non_optimized['Accuracy'],
                         normal_non_normal_non_optimized['Accuracy'],
                         batch_non_normal_non_optimized['Accuracy']], axis=1)

accuracy_df.columns = ['Online', 'Normal', 'Batch']
year_labels = ['Year_1', 'Year_2', 'Year_3', 'Year_4', 'Year_5', 'Year_6']

# Set the index of the DataFrame
accuracy_df.index = year_labels

mean_metrics_online_non_normal_non_optimized = accuracy_df['Online'].mean()
mean_metrics_normal_non_normal_non_optimized = accuracy_df['Normal'].mean()
mean_metrics_batch_non_normal_non_optimized = accuracy_df['Batch'].mean()

accuracy_df.plot(kind='line', figsize=(10, 6), grid=True)
plt.legend(labels=[f'Online (Avg: {mean_metrics_online_non_normal_non_optimized:.2f})',
                   f'Normal (Avg: {mean_metrics_normal_non_normal_non_optimized:.2f})',
                   f'Batch (Avg: {mean_metrics_batch_non_normal_non_optimized:.2f})'],
           loc='lower right')

plt.title('Accuracy Comparison')
plt.xlabel('Year')
plt.ylabel('Accuracy')
plt.ylim(ymin=0)
plt.show()

precision_df = pd.concat([online_non_normal_non_optimized['Precision'],
                         normal_non_normal_non_optimized['Precision'],
                         batch_non_normal_non_optimized['Precision']], axis=1)

precision_df.columns = ['Online', 'Normal', 'Batch']
# Set the index of the DataFrame
precision_df.index = year_labels

mean_metrics_online_non_normal_non_optimized = precision_df['Online'].mean()
mean_metrics_normal_non_normal_non_optimized = precision_df['Normal'].mean()
mean_metrics_batch_non_normal_non_optimized = precision_df['Batch'].mean()

precision_df.plot(kind='line', figsize=(10, 6), grid=True)
plt.legend(labels=[f'Online (Avg: {mean_metrics_online_non_normal_non_optimized:.2f})',
                   f'Normal (Avg: {mean_metrics_normal_non_normal_non_optimized:.2f})',
                   f'Batch (Avg: {mean_metrics_batch_non_normal_non_optimized:.2f})'],
           loc='lower right')

plt.title('Precision Comparison')
plt.xlabel('Year')
plt.ylabel('Precision')
plt.ylim(ymin=0)
plt.show()

recall_df = pd.concat([online_non_normal_non_optimized['Recall'],
                         normal_non_normal_non_optimized['Recall'],
                         batch_non_normal_non_optimized['Recall']], axis=1)

recall_df.columns = ['Online', 'Normal', 'Batch']
recall_df.index = year_labels

mean_metrics_online_non_normal_non_optimized = recall_df['Online'].mean()
mean_metrics_normal_non_normal_non_optimized = recall_df['Normal'].mean()
mean_metrics_batch_non_normal_non_optimized = recall_df['Batch'].mean()

recall_df.plot(kind='line', figsize=(10, 6), grid=True)
plt.legend(labels=[f'Online (Avg: {mean_metrics_online_non_normal_non_optimized:.2f})',
                   f'Normal (Avg: {mean_metrics_normal_non_normal_non_optimized:.2f})',
                   f'Batch (Avg: {mean_metrics_batch_non_normal_non_optimized:.2f})'],
           loc='lower right')

plt.title('Recall Comparison')
plt.xlabel('Year')
plt.ylabel('Recall')
plt.ylim(ymin=0)
plt.show()

f1_df = pd.concat([online_non_normal_non_optimized['F1'],
                         normal_non_normal_non_optimized['F1'],
                         batch_non_normal_non_optimized['F1']], axis=1)

f1_df.columns = ['Online', 'Normal', 'Batch']
f1_df.index = year_labels

mean_metrics_online_non_normal_non_optimized = f1_df['Online'].mean()
mean_metrics_normal_non_normal_non_optimized = f1_df['Normal'].mean()
mean_metrics_batch_non_normal_non_optimized = f1_df['Batch'].mean()

f1_df.plot(kind='line', figsize=(10, 6), grid=True)
plt.legend(labels=[f'Online (Avg: {mean_metrics_online_non_normal_non_optimized:.2f})',
                   f'Normal (Avg: {mean_metrics_normal_non_normal_non_optimized:.2f})',
                   f'Batch (Avg: {mean_metrics_batch_non_normal_non_optimized:.2f})'],
           loc='lower right')

plt.title('F1 Comparison')
plt.xlabel('Year')
plt.ylabel('F1')
plt.ylim(ymin=0)
plt.show()