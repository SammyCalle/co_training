import pickle

import keras
from keras.src.layers import Dense

with open("../../data/data25/both/not_balanced/both.pkl", "rb") as file:
    boths_25 = pickle.load(file)

with open("../../data/data25/systemcalls/not_balanced/system_calls.pkl", "rb") as file:
    system_calls_25 = pickle.load(file)

with open("../../data/data25/permissions/not_balanced/permissions.pkl", "rb") as file:
    permissions_25 = pickle.load(file)

with open("../../data/data25/labels/not_balanced/labels.pkl", "rb") as file:
    labels_25 = pickle.load(file)

with open("../../data/data75/both/not_balanced/both.pkl", "rb") as file:
    boths = pickle.load(file)

with open("../../data/data75/systemcalls/not_balanced/system_calls.pkl", "rb") as file:
    system_calls = pickle.load(file)

with open("../../data/data75/permissions/not_balanced/permissions.pkl", "rb") as file:
    permissions = pickle.load(file)

with open("../../data/data75/labels/not_balanced/labels.pkl", "rb") as file:
    labels = pickle.load(file)

with open("../../data/data50/both/not_balanced/both.pkl", "rb") as file:
    boths_50 = pickle.load(file)

with open("../../data/data50/systemcalls/not_balanced/system_calls.pkl", "rb") as file:
    system_calls_50 = pickle.load(file)

with open("../../data/data50/permissions/not_balanced/permissions.pkl", "rb") as file:
    permissions_50 = pickle.load(file)

with open("../../data/data50/labels/not_balanced/labels.pkl", "rb") as file:
    labels_50 = pickle.load(file)

clf_mlp_system_calls = keras.Sequential(
    [
        Dense(52, activation='relu', input_dim=287),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(128, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ]
)

clf_mlp_system_calls.compile(loss='binary_crossentropy', optimizer='adam')

clf_mlp_permissions = keras.Sequential(
    [
        Dense(52, activation='relu', input_dim=166),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ]
)

clf_mlp_permissions.compile(loss='binary_crossentropy', optimizer='adam')

clf_mlp_both = keras.Sequential(
    [
        Dense(52, activation='relu', input_dim=453),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ]
)

clf_mlp_both.compile(loss='binary_crossentropy', optimizer='adam')

model_dict_permissions = {}
for i, (permission, label) in enumerate(zip(permissions_25[:-1], labels_25[:-1])):
    clf_mlp_permissions.fit(permission, label)
    model_dict_permissions[i] = clf_mlp_permissions

with open("Normal/not_balanced/permissions/model25.pkl", "wb") as file:
    pickle.dump(model_dict_permissions, file)

model_dict_system_calls = {}
for i, (system_call, label) in enumerate(zip(system_calls_25[:-1], labels_25[:-1])):
    clf_mlp_system_calls.fit(system_call, label)
    model_dict_system_calls[i] = clf_mlp_system_calls

with open("Normal/not_balanced/systemcalls/model25.pkl", "wb") as file:
    pickle.dump(model_dict_system_calls, file)

model_dict_boths = {}
for i, (both, label) in enumerate(zip(boths_25[:-1], labels_25[:-1])):
    clf_mlp_both.fit(both, label)
    model_dict_boths[i] = clf_mlp_both

with open("Normal/not_balanced/both/model25.pkl", "wb") as file:
    pickle.dump(model_dict_boths, file)

model_dict_permissions = {}
for i, (permission, label) in enumerate(zip(permissions[:-1], labels[:-1])):
    clf_mlp_permissions.fit(permission, label)
    model_dict_permissions[i] = clf_mlp_permissions

with open("Normal/not_balanced/permissions/model75.pkl", "wb") as file:
    pickle.dump(model_dict_permissions, file)

model_dict_system_calls = {}
for i, (system_call, label) in enumerate(zip(system_calls[:-1], labels[:-1])):
    clf_mlp_system_calls.fit(system_call, label)
    model_dict_system_calls[i] = clf_mlp_system_calls

with open("Normal/not_balanced/systemcalls/model75.pkl", "wb") as file:
    pickle.dump(model_dict_system_calls, file)

model_dict_boths = {}
for i, (both, label) in enumerate(zip(boths[:-1], labels[:-1])):
    clf_mlp_both.fit(both, label)
    model_dict_boths[i] = clf_mlp_both

with open("Normal/not_balanced/both/model75.pkl", "wb") as file:
    pickle.dump(model_dict_boths, file)

model_dict_permissions = {}
for i, (permission, label) in enumerate(zip(permissions_50[:-1], labels_50[:-1])):
    clf_mlp_permissions.fit(permission, label)
    model_dict_permissions[i] = clf_mlp_permissions

with open("Normal/not_balanced/permissions/model50.pkl", "wb") as file:
    pickle.dump(model_dict_permissions, file)

model_dict_system_calls = {}
for i, (system_call, label) in enumerate(zip(system_calls_50[:-1], labels_50[:-1])):
    clf_mlp_system_calls.fit(system_call, label)
    model_dict_system_calls[i] = clf_mlp_system_calls

with open("Normal/not_balanced/systemcalls/model50.pkl", "wb") as file:
    pickle.dump(model_dict_system_calls, file)

model_dict_boths = {}
for i, (both, label) in enumerate(zip(boths_50[:-1], labels_50[:-1])):
    clf_mlp_both.fit(both, label)
    model_dict_boths[i] = clf_mlp_both

with open("Normal/not_balanced/both/model50.pkl", "wb") as file:
    pickle.dump(model_dict_boths, file)

