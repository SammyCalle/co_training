import pickle

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

with open("../data/data75/both/not_balanced/system_calls.pkl", "rb") as file:
    boths = pickle.load(file)

boths_batch = []
for i in range(len(boths)):
    if i != 0:
        boths_batch.append(pd.concat([boths[i - 1], boths[i]], ignore_index=True))
    else:
        boths_batch.append(boths[i])

with open("../data/data75/systemcalls/not_balanced/system_calls.pkl", "rb") as file:
    system_calls = pickle.load(file)

system_calls_batch = []
for i in range(len(boths)):
    if i != 0:
        system_calls_batch.append(pd.concat([system_calls[i - 1], system_calls[i]], ignore_index=True))
    else:
        system_calls_batch.append(system_calls[i])

with open("../data/data75/permissions/not_balanced/permissions.pkl", "rb") as file:
    permissions = pickle.load(file)

permissions_batch = []
for i in range(len(boths)):
    if i != 0:
        permissions_batch.append(pd.concat([permissions[i - 1], permissions[i]], ignore_index=True))
    else:
        permissions_batch.append(permissions[i])

with open("../data/data75/labels/not_balanced/labels.pkl", "rb") as file:
    labels = pickle.load(file)

labels_batch = []
for i in range(len(boths)):
    if i != 0:
        labels_batch.append(pd.concat([labels[i - 1], labels[i]], ignore_index=True))
    else:
        labels_batch.append(labels[i])

with open("../data/data50/both/not_balanced/system_calls.pkl", "rb") as file:
    boths_50 = pickle.load(file)

boths_50_batch = []
for i in range(len(boths)):
    if i != 0:
        boths_50_batch.append(pd.concat([boths_50[i - 1], boths_50[i]], ignore_index=True))
    else:
        boths_50_batch.append(boths_50[i])

with open("../data/data50/systemcalls/not_balanced/system_calls.pkl", "rb") as file:
    system_calls_50 = pickle.load(file)

system_calls_50_batch = []
for i in range(len(boths)):
    if i != 0:
        system_calls_50_batch.append(pd.concat([system_calls_50[i - 1], system_calls_50[i]], ignore_index=True))
    else:
        system_calls_50_batch.append(system_calls_50[i])

with open("../data/data50/permissions/not_balanced/permissions.pkl", "rb") as file:
    permissions_50 = pickle.load(file)

permissions_50_batch = []
for i in range(len(boths)):
    if i != 0:
        permissions_50_batch.append(pd.concat([permissions_50[i - 1], permissions_50[i]], ignore_index=True))
    else:
        permissions_50_batch.append(permissions_50[i])

with open("../data/data50/labels/not_balanced/labels.pkl", "rb") as file:
    labels_50 = pickle.load(file)

labels_50_batch = []
for i in range(len(boths)):
    if i != 0:
        labels_50_batch.append(pd.concat([labels_50[i - 1], labels_50[i]], ignore_index=True))
    else:
        labels_50_batch.append(labels_50[i])

model_system_calls = RandomForestClassifier(max_features=None, n_estimators=200)

model_permissions = RandomForestClassifier(max_features='log2', n_estimators=50)

model_both = RandomForestClassifier(n_estimators=150)

model_dict_permissions = {}
for i, (permission, label) in enumerate(zip(permissions[:-1], labels[:-1])):
    model_permissions.fit(permission, label)
    model_dict_permissions[i] = model_permissions

with open("Batch/not_balanced/permissions/model75.pkl", "wb") as file:
    pickle.dump(model_dict_permissions, file)

model_dict_system_calls = {}
for i, (system_call, label) in enumerate(zip(system_calls[:-1], labels[:-1])):
    model_system_calls.fit(system_call, label)
    model_dict_system_calls[i] = model_system_calls

with open("Batch/not_balanced/systemcalls/model75.pkl", "wb") as file:
    pickle.dump(model_dict_system_calls, file)

model_dict_boths = {}
for i, (both, label) in enumerate(zip(boths[:-1], labels[:-1])):
    model_both.fit(both, label)
    model_dict_boths[i] = model_both

with open("Batch/not_balanced/both/model75.pkl", "wb") as file:
    pickle.dump(model_dict_boths, file)

model_dict_permissions = {}
for i, (permission, label) in enumerate(zip(permissions_50[:-1], labels_50[:-1])):
    model_permissions.fit(permission, label)
    model_dict_permissions[i] = model_permissions

with open("Batch/not_balanced/permissions/model50.pkl", "wb") as file:
    pickle.dump(model_dict_permissions, file)

model_dict_system_calls = {}
for i, (system_call, label) in enumerate(zip(system_calls_50[:-1], labels_50[:-1])):
    model_system_calls.fit(system_call, label)
    model_dict_system_calls[i] = model_system_calls

with open("Batch/not_balanced/systemcalls/model50.pkl", "wb") as file:
    pickle.dump(model_dict_system_calls, file)

model_dict_boths = {}
for i, (both, label) in enumerate(zip(boths_50[:-1], labels_50[:-1])):
    model_both.fit(both, label)
    model_dict_boths[i] = model_both

with open("Batch/not_balanced/both/model50.pkl", "wb") as file:
    pickle.dump(model_dict_boths, file)

