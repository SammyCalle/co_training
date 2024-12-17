import pickle

from river import forest, stream
import pandas as pd
import numpy as np

with open("../../data/data25/both/not_balanced/both.pkl", "rb") as file:
    boths_25 = pickle.load(file)

with open("../../data/data25/systemcalls/not_balanced/system_calls.pkl", "rb") as file:
    system_calls_25 = pickle.load(file)

with open("../../data/data25/permissions/not_balanced/permissions.pkl", "rb") as file:
    permissions_25 = pickle.load(file)

with open("../../data/data25/labels/not_balanced/labels.pkl", "rb") as file:
    labels_25 = pickle.load(file)

# with open("../../data/data75/both/balanced/both.pkl", "rb") as file:
#     boths = pickle.load(file)
#
# with open("../../data/data75/systemcalls/balanced/system_calls.pkl", "rb") as file:
#     system_calls = pickle.load(file)
#
# with open("../../data/data75/permissions/balanced/permissions.pkl", "rb") as file:
#     permissions = pickle.load(file)
#
# with open("../../data/data75/labels/balanced/labels.pkl", "rb") as file:
#     labels = pickle.load(file)
#
# with open("../../data/data50/both/balanced/both.pkl", "rb") as file:
#     boths_50 = pickle.load(file)
#
# with open("../../data/data50/systemcalls/balanced/system_calls.pkl", "rb") as file:
#     system_calls_50 = pickle.load(file)
#
# with open("../../data/data50/permissions/balanced/permissions.pkl", "rb") as file:
#     permissions_50 = pickle.load(file)
#
# with open("../../data/data50/labels/balanced/labels.pkl", "rb") as file:
#     labels_50 = pickle.load(file)

model_system_calls = forest.ARFClassifier(n_models=20)

model_permissions = forest.ARFClassifier(n_models=30)

model_both = forest.ARFClassifier(n_models=20)

model_dict_permissions = {}
for i, (permission, label) in enumerate(zip(permissions_25[:6], labels_25[:6])):
    stream_data = stream.iter_pandas(permission, label)
    for x, y in stream_data:
        model_permissions.learn_one(x, y)
    model_dict_permissions[i] = model_permissions

with open("not_balanced/permissions/model25.pkl", "wb") as file:
    pickle.dump(model_dict_permissions, file)

model_dict_system_calls = {}
for i, (system_call, label) in enumerate(zip(system_calls_25[:6], labels_25[:6])):
    stream_data = stream.iter_pandas(system_call, label)
    for x, y in stream_data:
        model_system_calls.learn_one(x, y)
    model_dict_system_calls[i] = model_system_calls

with open("not_balanced/systemcalls/model25.pkl", "wb") as file:
    pickle.dump(model_dict_system_calls, file)

model_dict_boths = {}
for i, (both, label) in enumerate(zip(boths_25[:6], labels_25[:6])):
    stream_data = stream.iter_pandas(both, label)
    for x, y in stream_data:
        model_both.learn_one(x, y)
    model_dict_boths[i] = model_both

with open("not_balanced/both/model25.pkl", "wb") as file:
    pickle.dump(model_dict_boths, file)
#
# model_dict_permissions = {}
# for i, (permission, label) in enumerate(zip(permissions[:6], labels[:6])):
#     stream_data = stream.iter_pandas(permission, label)
#     for x, y in stream_data:
#         model_permissions.learn_one(x, y)
#     model_dict_permissions[i] = model_permissions
#
# with open("balanced/permissions/model75.pkl", "wb") as file:
#     pickle.dump(model_dict_permissions, file)
#
# model_dict_system_calls = {}
# for i, (system_call, label) in enumerate(zip(system_calls[:6], labels[:6])):
#     stream_data = stream.iter_pandas(system_call, label)
#     for x, y in stream_data:
#         model_system_calls.learn_one(x, y)
#     model_dict_system_calls[i] = model_system_calls
#
# with open("balanced/systemcalls/model75.pkl", "wb") as file:
#     pickle.dump(model_dict_system_calls, file)
#
# model_dict_boths = {}
# for i, (both, label) in enumerate(zip(boths[:6], labels[:6])):
#     stream_data = stream.iter_pandas(both, label)
#     for x, y in stream_data:
#         model_both.learn_one(x, y)
#     model_dict_boths[i] = model_both
#
# with open("balanced/both/model75.pkl", "wb") as file:
#     pickle.dump(model_dict_boths, file)
#
# model_dict_permissions = {}
# for i, (permission, label) in enumerate(zip(permissions_50[:6], labels_50[:6])):
#     stream_data = stream.iter_pandas(permission, label)
#     for x, y in stream_data:
#         model_permissions.learn_one(x, y)
#     model_dict_permissions[i] = model_permissions
#
# with open("balanced/permissions/model50.pkl", "wb") as file:
#     pickle.dump(model_dict_permissions, file)
#
# model_dict_system_calls = {}
# for i, (system_call, label) in enumerate(zip(system_calls_50[:6], labels_50[:6])):
#     stream_data = stream.iter_pandas(system_call, label)
#     for x, y in stream_data:
#         model_system_calls.learn_one(x, y)
#     model_dict_system_calls[i] = model_system_calls
#
# with open("balanced/systemcalls/model50.pkl", "wb") as file:
#     pickle.dump(model_dict_system_calls, file)
#
# model_dict_boths = {}
# for i, (both, label) in enumerate(zip(boths_50[:6], labels_50[:6])):
#     stream_data = stream.iter_pandas(both, label)
#     for x, y in stream_data:
#         model_both.learn_one(x, y)
#     model_dict_boths[i] = model_both
#
# with open("balanced/both/model50.pkl", "wb") as file:
#     pickle.dump(model_dict_boths, file)
#
