import pickle

from sklearn.model_selection import train_test_split

with open("data/data.pkl", 'rb') as f:
    data = pickle.load(f)

X = data.drop("Malware", axis=1)
X_system_calls = X.iloc[:, 2:289]
X_permissions = X.iloc[:, 289:]
y = data['Malware']

X_train, X_test, \
    y_train, y_test = train_test_split(X, y,
                                       test_size=0.25,
                                       random_state=42)

# X_train_system_calls, X_test_system_calls,\
#     y_train_system_calls, y_test_system_calls = train_test_split(X_system_calls, y,
#                                        test_size=0.25,
#                                        random_state=42)
#
# X_train_permissions, X_test_permissions,\
#     y_train_permissions, y_test_permissions = train_test_split(X_permissions, y,
#                                        test_size=0.25,
#                                        random_state=42)

with open("data/both/X_train.pkl", "wb") as file:
    pickle.dump(X_train, file)

with open("data/both/X_test.pkl", "wb") as file:
    pickle.dump(X_test, file)

with open("data/both/y_train.pkl", "wb") as file:
    pickle.dump(y_train, file)

with open("data/both/y_test.pkl", "wb") as file:
    pickle.dump(y_test, file)

# with open("data/system_calls/X_train_system_calls.pkl", "wb") as file:
#     pickle.dump(X_train_system_calls, file)
#
# with open("data/system_calls/X_test_system_calls.pkl", "wb") as file:
#     pickle.dump(X_test_system_calls, file)
#
# with open("data/system_calls/y_train_system_calls.pkl", "wb") as file:
#     pickle.dump(y_train_system_calls, file)
#
# with open("data/system_calls/y_test_system_calls.pkl", "wb") as file:
#     pickle.dump(y_test_system_calls, file)
#
# with open("data/permissions/X_train_permissions.pkl", "wb") as file:
#     pickle.dump(X_train_permissions, file)
#
# with open("data/permissions/X_test_permissions.pkl", "wb") as file:
#     pickle.dump(X_test_permissions, file)
#
# with open("data/permissions/y_train_permissions.pkl", "wb") as file:
#     pickle.dump(y_train_permissions, file)
#
# with open("data/permissions/y_test_permissions.pkl", "wb") as file:
#     pickle.dump(y_test_permissions, file)
