from imblearn.over_sampling import SMOTE
import pandas as pd

def cotraining_preparation(data_list, resample):

    if resample:
        for i, data in enumerate(data_list):
            y = data['Malware']
            X = data.drop('Malware', axis=1)

            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            data_resampled = pd.concat([X_resampled, y_resampled], axis=1)
            data_list[i] = data_resampled

    system_calls = []
    permissions = []
    y = []

    for data in data_list:
        # Get the number of rows in the DataFrame

        num_rows = len(data)
        # Calculate the number of samples to change
        num_to_change = int(num_rows * 0.25)

        data.loc[num_to_change:, 'Malware'] = -1

    for data in data_list:
        system_calls.append(data.iloc[:, 2:289])
        permissions.append(data.iloc[:, 289:])
        y.append(data.pop('Malware'))

    return system_calls, permissions, y
