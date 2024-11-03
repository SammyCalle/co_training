def SplitViews(data_list):

    system_calls = []
    permissions = []
    y = []

    for data in data_list:
        system_calls.append(data.iloc[:, 2:289])
        permissions.append(data.iloc[:, 289:])
        y.append(data.pop('Malware'))

    return system_calls, permissions, y