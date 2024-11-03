import pandas as pd
import numpy as np


def cotraining_preparation(data_list):

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
