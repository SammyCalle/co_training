import pickle

from river import forest,stream
import numpy as np

with open("../data/data75/both/not_balanced/system_calls.pkl", "rb") as file:
    boths = pickle.load(file)

with open("../data/data75/systemcalls/not_balanced/system_calls.pkl", "rb") as file:
    system_calls = pickle.load(file)

with open("../data/data75/permissions/not_balanced/permissions.pkl", "rb") as file:
    permissions = pickle.load(file)

with open("../data/data75/labels/not_balanced/labels.pkl", "rb") as file:
    labels = pickle.load(file)

with open("../data/data50/both/not_balanced/system_calls.pkl", "rb") as file:
    boths_50 = pickle.load(file)

with open("../data/data50/systemcalls/not_balanced/system_calls.pkl", "rb") as file:
    system_calls_50 = pickle.load(file)

with open("../data/data50/permissions/not_balanced/permissions.pkl", "rb") as file:
    permissions_50 = pickle.load(file)

with open("../data/data50/labels/not_balanced/labels.pkl", "rb") as file:
    labels_50 = pickle.load(file)

model_system_calls = forest.ARFClassifier(max_features=287,n_models=200)

model_permissions = forest.ARFClassifier(max_features=int(np.log2(166)),n_models=50)

model_both = forest.ARFClassifier(max_features=453,n_models=150)

# model_dict_permissions = {}
# for i, (permission, label) in enumerate(zip(permissions, labels)):
#     stream_data = stream.iter_pandas(permission, label)
#     for x, y in stream_data:
#         model_permissions.learn_one(x, y)
#     model_dict_permissions[i] = model_permissions
#
# with open("not_balanced/permissions/model75.pkl", "wb") as file:
#     pickle.dump(model_dict_permissions, file)

model_dict_system_calls = {}
for i, (system_call, label) in enumerate(zip(system_calls, labels)):
    stream_data = stream.iter_pandas(system_call, label)
    for x, y in stream_data:
        model_system_calls.learn_one(x, y)
    model_dict_system_calls[i] = model_system_calls

with open("not_balanced/systemcalls/model75.pkl", "wb") as file:
    pickle.dump(model_dict_system_calls, file)

# model_dict_boths = {}
# for i, (both, label) in enumerate(zip(boths, labels)):
#     stream_data = stream.iter_pandas(both, label)
#     for x, y in stream_data:
#         model_both.learn_one(x, y)
#     model_dict_boths[i] = model_both
#
# with open("not_balanced/both/model75.pkl", "wb") as file:
#     pickle.dump(model_dict_boths, file)
#
# model_dict_permissions = {}
# for i, (permission, label) in enumerate(zip(permissions_50, labels_50)):
#     stream_data = stream.iter_pandas(permission, label)
#     for x, y in stream_data:
#         model_permissions.learn_one(x, y)
#     model_dict_permissions[i] = model_permissions
#
# with open("not_balanced/permissions/model50.pkl", "wb") as file:
#     pickle.dump(model_dict_permissions, file)
#
# model_dict_system_calls = {}
# for i, (system_call, label) in enumerate(zip(system_calls_50, labels_50)):
#     stream_data = stream.iter_pandas(system_call, label)
#     for x, y in stream_data:
#         model_system_calls.learn_one(x, y)
#     model_dict_system_calls[i] = model_system_calls
#
# with open("not_balanced/systemcalls/model50.pkl", "wb") as file:
#     pickle.dump(model_dict_system_calls, file)
#
# model_dict_boths = {}
# for i, (both, label) in enumerate(zip(boths_50, labels_50)):
#     stream_data = stream.iter_pandas(both, label)
#     for x, y in stream_data:
#         model_both.learn_one(x, y)
#     model_dict_boths[i] = model_both
#
# with open("not_balanced/both/model50.pkl", "wb") as file:
#     pickle.dump(model_dict_boths, file)

