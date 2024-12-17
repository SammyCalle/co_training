import pickle

from imblearn.over_sampling import SMOTE
import pandas as pd

with open("../../DataProcessing/YearData/FullData/fulldata_year.pkl", 'rb') as f:
    data_list = pickle.load(f)


def split_data(data_list, resample, percentage_data):
    assert (0 < percentage_data <= 1)
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
    system_calls_and_permissions = []
    y = []

    for data in data_list:
        # Get the number of rows in the DataFrame

        data = data.sample(frac=percentage_data)

        y.append(data.pop('Malware'))
        system_calls_and_permissions.append(data.iloc[:, 1:])
        system_calls.append(data.iloc[:, 1:288])
        permissions.append(data.iloc[:, 288:])

    return system_calls_and_permissions, system_calls, permissions, y


system_calls_and_permissions, system_calls, permissions, labels = split_data(data_list=data_list, resample=False,
                                                                             percentage_data=0.25)

with open("data25/both/not_balanced/both.pkl", "wb") as file:
    pickle.dump(system_calls_and_permissions, file)

with open("data25/systemcalls/not_balanced/system_calls.pkl", "wb") as file:
    pickle.dump(system_calls, file)

with open("data25/permissions/not_balanced/permissions.pkl", "wb") as file:
    pickle.dump(permissions, file)

with open("data25/labels/not_balanced/labels.pkl", "wb") as file:
    pickle.dump(labels, file)
