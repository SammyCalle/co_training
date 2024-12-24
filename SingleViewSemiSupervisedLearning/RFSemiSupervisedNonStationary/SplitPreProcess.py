def SplitViews(data_list):

    both = []
    system_calls = []
    permissions = []
    y = []

    for data in data_list[1:]:
        both.append(data.iloc[:, 2:])
        system_calls.append(data.iloc[:, 2:289])
        permissions.append(data.iloc[:, 289:])
        y.append(data.pop('Malware'))

    return both, system_calls, permissions, y