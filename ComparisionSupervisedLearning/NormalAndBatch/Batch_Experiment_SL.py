import pickle

from sklearn.ensemble import RandomForestClassifier

with open("../data/data75/both/balanced/system_calls.pkl", "rb") as file:
    both = pickle.load(file)

with open("../data/data75/systemcalls/balanced/system_calls.pkl", "rb") as file:
    system_calls = pickle.load(file)

with open("../data/data75/permissions/balanced/permissions.pkl", "rb") as file:
    system_calls = pickle.load(file)

with open("../data/data75/labels/balanced/labels.pkl", "rb") as file:
    system_calls = pickle.load(file)


model_system_calls = RandomForestClassifier(max_features=None, n_estimators=200)
model_permissions = RandomForestClassifier(max_features='log2', n_estimators=50)
