import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

with open("../Online/YearExperimentResults/non_normal/metrics.pkl", 'rb') as f:
    online_non_normal_non_optimized = pickle.load(f)

with open("../NormalAndBatch/YearExperimentResults/Normal/non_normal/metrics.pkl", 'rb') as f:
    normal_non_normal_non_optimized = pickle.load(f)

with open("../NormalAndBatch/YearExperimentResults/Batch/non_normal/metrics.pkl", 'rb') as f:
    batch_non_normal_non_optimized = pickle.load(f)


accuracy_df = pd.concat([online_non_normal_non_optimized['Accuracy'],
                         normal_non_normal_non_optimized['Accuracy'],
                         batch_non_normal_non_optimized['Accuracy']], axis=1)

accuracy_df.columns = ['Online', 'Normal', 'Batch']

mean_metrics_online_non_normal_non_optimized = accuracy_df['Online'].mean()
mean_metrics_normal_non_normal_non_optimized = accuracy_df['Normal'].mean()
mean_metrics_batch_non_normal_non_optimized = accuracy_df['Batch'].mean()

accuracy_df.plot(kind='line', figsize=(10, 6), grid=True)
plt.legend(labels=[f'Online (Avg: {mean_metrics_online_non_normal_non_optimized:.2f})',
                   f'Normal (Avg: {mean_metrics_normal_non_normal_non_optimized:.2f})',
                   f'Batch (Avg: {mean_metrics_batch_non_normal_non_optimized:.2f})'],
           loc='lower right')

plt.title('Accuracy Comparison')
plt.xlabel('Experiment')
plt.ylabel('Accuracy')
plt.ylim(ymin=0)
plt.show()

accuracy_df = pd.concat([online_non_normal_non_optimized['Precision'],
                         normal_non_normal_non_optimized['Precision'],
                         batch_non_normal_non_optimized['Precision']], axis=1)

accuracy_df.columns = ['Online', 'Normal', 'Batch']

mean_metrics_online_non_normal_non_optimized = accuracy_df['Online'].mean()
mean_metrics_normal_non_normal_non_optimized = accuracy_df['Normal'].mean()
mean_metrics_batch_non_normal_non_optimized = accuracy_df['Batch'].mean()

accuracy_df.plot(kind='line', figsize=(10, 6), grid=True)
plt.legend(labels=[f'Online (Avg: {mean_metrics_online_non_normal_non_optimized:.2f})',
                   f'Normal (Avg: {mean_metrics_normal_non_normal_non_optimized:.2f})',
                   f'Batch (Avg: {mean_metrics_batch_non_normal_non_optimized:.2f})'],
           loc='lower right')

plt.title('Precision Comparison')
plt.xlabel('Experiment')
plt.ylabel('Precision')
plt.ylim(ymin=0)
plt.show()

accuracy_df = pd.concat([online_non_normal_non_optimized['Recall'],
                         normal_non_normal_non_optimized['Recall'],
                         batch_non_normal_non_optimized['Recall']], axis=1)

accuracy_df.columns = ['Online', 'Normal', 'Batch']

mean_metrics_online_non_normal_non_optimized = accuracy_df['Online'].mean()
mean_metrics_normal_non_normal_non_optimized = accuracy_df['Normal'].mean()
mean_metrics_batch_non_normal_non_optimized = accuracy_df['Batch'].mean()

accuracy_df.plot(kind='line', figsize=(10, 6), grid=True)
plt.legend(labels=[f'Online (Avg: {mean_metrics_online_non_normal_non_optimized:.2f})',
                   f'Normal (Avg: {mean_metrics_normal_non_normal_non_optimized:.2f})',
                   f'Batch (Avg: {mean_metrics_batch_non_normal_non_optimized:.2f})'],
           loc='lower right')

plt.title('Recall Comparison')
plt.xlabel('Experiment')
plt.ylabel('Recall')
plt.ylim(ymin=0)
plt.show()

accuracy_df = pd.concat([online_non_normal_non_optimized['F1'],
                         normal_non_normal_non_optimized['F1'],
                         batch_non_normal_non_optimized['F1']], axis=1)

accuracy_df.columns = ['Online', 'Normal', 'Batch']

mean_metrics_online_non_normal_non_optimized = accuracy_df['Online'].mean()
mean_metrics_normal_non_normal_non_optimized = accuracy_df['Normal'].mean()
mean_metrics_batch_non_normal_non_optimized = accuracy_df['Batch'].mean()

accuracy_df.plot(kind='line', figsize=(10, 6), grid=True)
plt.legend(labels=[f'Online (Avg: {mean_metrics_online_non_normal_non_optimized:.2f})',
                   f'Normal (Avg: {mean_metrics_normal_non_normal_non_optimized:.2f})',
                   f'Batch (Avg: {mean_metrics_batch_non_normal_non_optimized:.2f})'],
           loc='lower right')

plt.title('F1 Comparison')
plt.xlabel('Experiment')
plt.ylabel('F1')
plt.ylim(ymin=0)
plt.show()