import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

with open("../RamdonForest/NormalAndBatch/Normal/balanced/permissions/metrics50.pkl", 'rb') as f:
    normal_balanced_permissions_50 = pickle.load(f)

with open("../RamdonForest/NormalAndBatch/Normal/balanced/systemcalls/metrics50.pkl", 'rb') as f:
    normal_balanced_system_calls_50 = pickle.load(f)

with open("../RamdonForest/NormalAndBatch/Normal/balanced/both/metrics50.pkl", 'rb') as f:
    normal_balanced_both_50 = pickle.load(f)

with open("../RamdonForest/NormalAndBatch/Normal/balanced/permissions/metrics75.pkl", 'rb') as f:
    normal_balanced_permissions_75 = pickle.load(f)

with open("../RamdonForest/NormalAndBatch/Normal/balanced/systemcalls/metrics75.pkl", 'rb') as f:
    normal_balanced_system_calls_75 = pickle.load(f)

with open("../RamdonForest/NormalAndBatch/Normal/balanced/both/metrics75.pkl", 'rb') as f:
    normal_balanced_both_75 = pickle.load(f)

with open(
        "../../RFCotrainingNonStationary/NormalAndBatch/YearExperimentResults/Normal/non_normal/optimized/OnlyModel/balanced/metrics.pkl",
        'rb') as f:
    coTraining = pickle.load(f)

# Accuracy Plot-----------------------------------------------------------------------------------------
accuracy_df = pd.concat([normal_balanced_permissions_50['Accuracy'],
                         normal_balanced_system_calls_50['Accuracy'],
                         normal_balanced_both_50['Accuracy'],
                         normal_balanced_permissions_75['Accuracy'],
                         normal_balanced_system_calls_75['Accuracy'],
                         normal_balanced_both_75['Accuracy'],
                         coTraining['Accuracy']], axis=1)

accuracy_df.columns = ['Permission_50', 'SystemCalls_50', 'Both_50', 'Permission_75', 'SystemCalls_75', 'Both_75',
                       'CoTraining']
year_labels = ['Year_1', 'Year_2', 'Year_3', 'Year_4', 'Year_5', 'Year_6']

# Set the index of the DataFrame
accuracy_df.index = year_labels

mean_metrics_normal_balanced_permissions_50 = accuracy_df['Permission_50'].mean()
mean_metrics_normal_balanced_system_calls_50 = accuracy_df['SystemCalls_50'].mean()
mean_metrics_normal_balanced_both_50 = accuracy_df['Both_50'].mean()
mean_metrics_normal_balanced_permissions_75 = accuracy_df['Permission_75'].mean()
mean_metrics_normal_balanced_system_calls_75 = accuracy_df['SystemCalls_75'].mean()
mean_metrics_normal_balanced_both_75 = accuracy_df['Both_75'].mean()
mean_metrics_coTraining = accuracy_df['CoTraining'].mean()

accuracy_df.plot(kind='line', figsize=(10, 6), grid=True)
plt.legend(labels=[f'Permissions 50 (Avg: {mean_metrics_normal_balanced_permissions_50:.2f})',
                   f'SystemCalls 50 (Avg: {mean_metrics_normal_balanced_system_calls_50:.2f})',
                   f'Both 50 (Avg: {mean_metrics_normal_balanced_both_50:.2f})',
                   f'Permissions 75 (Avg: {mean_metrics_normal_balanced_permissions_75:.2f})',
                   f'SystemCalls 75 (Avg: {mean_metrics_normal_balanced_system_calls_75:.2f})',
                   f'Both 75 (Avg: {mean_metrics_normal_balanced_both_75:.2f})',
                   f'CoTraining(Avg: {mean_metrics_coTraining:.2f})'],
           loc='lower right')

plt.title('Accuracy Comparison')
plt.xlabel('Year')
plt.ylabel('Accuracy')
plt.ylim(ymin=0)
plt.show()

# Precision Plot-----------------------------------------------------------------------------------------

precision_df = pd.concat([normal_balanced_permissions_50['Precision'],
                          normal_balanced_system_calls_50['Precision'],
                          normal_balanced_both_50['Precision'],
                          normal_balanced_permissions_75['Precision'],
                          normal_balanced_system_calls_75['Precision'],
                          normal_balanced_both_75['Precision'],
                          coTraining['Precision']], axis=1)

precision_df.columns = ['Permission_50', 'SystemCalls_50', 'Both_50', 'Permission_75', 'SystemCalls_75', 'Both_75',
                        'CoTraining']
# Set the index of the DataFrame
precision_df.index = year_labels

mean_metrics_normal_balanced_permissions_50 = precision_df['Permission_50'].mean()
mean_metrics_normal_balanced_system_calls_50 = precision_df['SystemCalls_50'].mean()
mean_metrics_normal_balanced_both_50 = precision_df['Both_50'].mean()
mean_metrics_normal_balanced_permissions_75 = precision_df['Permission_75'].mean()
mean_metrics_normal_balanced_system_calls_75 = precision_df['SystemCalls_75'].mean()
mean_metrics_normal_balanced_both_75 = precision_df['Both_75'].mean()
mean_metrics_coTraining = precision_df['CoTraining'].mean()

precision_df.plot(kind='line', figsize=(10, 6), grid=True)
plt.legend(labels=[f'Permissions 50 (Avg: {mean_metrics_normal_balanced_permissions_50:.2f})',
                   f'SystemCalls 50 (Avg: {mean_metrics_normal_balanced_system_calls_50:.2f})',
                   f'Both 50 (Avg: {mean_metrics_normal_balanced_both_50:.2f})',
                   f'Permissions 75 (Avg: {mean_metrics_normal_balanced_permissions_75:.2f})',
                   f'SystemCals 75 (Avg: {mean_metrics_normal_balanced_system_calls_75:.2f})',
                   f'Both 75 (Avg: {mean_metrics_normal_balanced_both_75:.2f})',
                   f'CoTraining(Avg: {mean_metrics_coTraining:.2f})'],
           loc='lower right')

plt.title('Precision Comparison')
plt.xlabel('Year')
plt.ylabel('Precision')
plt.ylim(ymin=0)
plt.show()

# Recall Plot-----------------------------------------------------------------------------------------
recall_df = pd.concat([normal_balanced_permissions_50['Recall'],
                       normal_balanced_system_calls_50['Recall'],
                       normal_balanced_both_50['Recall'],
                       normal_balanced_permissions_75['Recall'],
                       normal_balanced_system_calls_75['Recall'],
                       normal_balanced_both_75['Recall'],
                       coTraining['Recall']], axis=1)

recall_df.columns = ['Permission_50', 'SystemCalls_50', 'Both_50', 'Permission_75', 'SystemCalls_75', 'Both_75',
                     'CoTraining']
recall_df.index = year_labels

mean_metrics_normal_balanced_permissions_50 = recall_df['Permission_50'].mean()
mean_metrics_normal_balanced_system_calls_50 = recall_df['SystemCalls_50'].mean()
mean_metrics_normal_balanced_both_50 = recall_df['Both_50'].mean()
mean_metrics_normal_balanced_permissions_75 = recall_df['Permission_75'].mean()
mean_metrics_normal_balanced_system_calls_75 = recall_df['SystemCalls_75'].mean()
mean_metrics_normal_balanced_both_75 = recall_df['Both_75'].mean()
mean_metrics_coTraining = recall_df['CoTraining'].mean()

recall_df.plot(kind='line', figsize=(10, 6), grid=True)
plt.legend(labels=[f'Permissions 50 (Avg: {mean_metrics_normal_balanced_permissions_50:.2f})',
                   f'SystemCalls 50 (Avg: {mean_metrics_normal_balanced_system_calls_50:.2f})',
                   f'Both 50 (Avg: {mean_metrics_normal_balanced_both_50:.2f})',
                   f'Permissions 75 (Avg: {mean_metrics_normal_balanced_permissions_75:.2f})',
                   f'SystemCals 75 (Avg: {mean_metrics_normal_balanced_system_calls_75:.2f})',
                   f'Both 75 (Avg: {mean_metrics_normal_balanced_both_75:.2f})',
                   f'CoTraining(Avg: {mean_metrics_coTraining:.2f})'],
           loc='lower right')

plt.title('Recall Comparison')
plt.xlabel('Year')
plt.ylabel('Recall')
plt.ylim(ymin=0)
plt.show()

# F1 Plot-----------------------------------------------------------------------------------------

f1_df = pd.concat([normal_balanced_permissions_50['F1'],
                   normal_balanced_system_calls_50['F1'],
                   normal_balanced_both_50['F1'],
                   normal_balanced_permissions_75['F1'],
                   normal_balanced_system_calls_75['F1'],
                   normal_balanced_both_75['F1'],
                   coTraining['F1']], axis=1)

f1_df.columns = ['Permission_50', 'SystemCalls_50', 'Both_50', 'Permission_75', 'SystemCalls_75', 'Both_75',
                 'CoTraining']
f1_df.index = year_labels

mean_metrics_normal_balanced_permissions_50 = f1_df['Permission_50'].mean()
mean_metrics_normal_balanced_system_calls_50 = f1_df['SystemCalls_50'].mean()
mean_metrics_normal_balanced_both_50 = f1_df['Both_50'].mean()
mean_metrics_normal_balanced_permissions_75 = f1_df['Permission_75'].mean()
mean_metrics_normal_balanced_system_calls_75 = f1_df['SystemCalls_75'].mean()
mean_metrics_normal_balanced_both_75 = f1_df['Both_75'].mean()
mean_metrics_coTraining = f1_df['CoTraining'].mean()

f1_df.plot(kind='line', figsize=(10, 6), grid=True)
plt.legend(labels=[f'Permissions 50 (Avg: {mean_metrics_normal_balanced_permissions_50:.2f})',
                   f'SystemCalls 50 (Avg: {mean_metrics_normal_balanced_system_calls_50:.2f})',
                   f'Both 50 (Avg: {mean_metrics_normal_balanced_both_50:.2f})',
                   f'Permissions 75 (Avg: {mean_metrics_normal_balanced_permissions_75:.2f})',
                   f'SystemCals 75 (Avg: {mean_metrics_normal_balanced_system_calls_75:.2f})',
                   f'Both 75 (Avg: {mean_metrics_normal_balanced_both_75:.2f})',
                   f'CoTraining(Avg: {mean_metrics_coTraining:.2f})'],
           loc='lower right')

plt.title('F1 Comparison')
plt.xlabel('Year')
plt.ylabel('F1')
plt.ylim(ymin=0)
plt.show()
