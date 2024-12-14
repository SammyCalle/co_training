import pickle

from sklearn.ensemble import RandomForestClassifier

with open("../../data/data75/both/balanced/both.pkl", "rb") as file:
    boths = pickle.load(file)

with open("../../data/data75/systemcalls/balanced/system_calls.pkl", "rb") as file:
    system_calls = pickle.load(file)

with open("../../data/data75/permissions/balanced/permissions.pkl", "rb") as file:
    permissions = pickle.load(file)

with open("../../data/data75/labels/balanced/labels.pkl", "rb") as file:
    labels = pickle.load(file)

with open("../../data/data50/both/balanced/both.pkl", "rb") as file:
    boths_50 = pickle.load(file)

with open("../../data/data50/systemcalls/balanced/system_calls.pkl", "rb") as file:
    system_calls_50 = pickle.load(file)

with open("../../data/data50/permissions/balanced/permissions.pkl", "rb") as file:
    permissions_50 = pickle.load(file)

with open("../../data/data50/labels/balanced/labels.pkl", "rb") as file:
    labels_50 = pickle.load(file)

model_system_calls = RandomForestClassifier(max_features=None, n_estimators=200)

model_permissions = RandomForestClassifier(max_features='log2', n_estimators=50)

model_both = RandomForestClassifier(n_estimators=150)

model_dict_permissions = {}
for i, (permission, label) in enumerate(zip(permissions[:6], labels[:6])):
    model_permissions.fit(permission, label)
    model_dict_permissions[i] = model_permissions

with open("Normal/balanced/permissions/model75.pkl", "wb") as file:
    pickle.dump(model_dict_permissions, file)

model_dict_system_calls = {}
for i, (system_call, label) in enumerate(zip(system_calls[:6], labels[:6])):
    model_system_calls.fit(system_call, label)
    model_dict_system_calls[i] = model_system_calls

with open("Normal/balanced/systemcalls/model75.pkl", "wb") as file:
    pickle.dump(model_dict_system_calls, file)

model_dict_boths = {}
for i, (both, label) in enumerate(zip(boths[:6], labels[:6])):
    model_both.fit(both, label)
    model_dict_boths[i] = model_both

with open("Normal/balanced/both/model75.pkl", "wb") as file:
    pickle.dump(model_dict_boths, file)

model_dict_permissions = {}
for i, (permission, label) in enumerate(zip(permissions_50[:6], labels_50[:6])):
    model_permissions.fit(permission, label)
    model_dict_permissions[i] = model_permissions

with open("Normal/balanced/permissions/model50.pkl", "wb") as file:
    pickle.dump(model_dict_permissions, file)

model_dict_system_calls = {}
for i, (system_call, label) in enumerate(zip(system_calls_50[:6], labels_50[:6])):
    model_system_calls.fit(system_call, label)
    model_dict_system_calls[i] = model_system_calls

with open("Normal/balanced/systemcalls/model50.pkl", "wb") as file:
    pickle.dump(model_dict_system_calls, file)

model_dict_boths = {}
for i, (both, label) in enumerate(zip(boths_50[:6], labels_50[:6])):
    model_both.fit(both, label)
    model_dict_boths[i] = model_both

with open("Normal/balanced/both/model50.pkl", "wb") as file:
    pickle.dump(model_dict_boths, file)

