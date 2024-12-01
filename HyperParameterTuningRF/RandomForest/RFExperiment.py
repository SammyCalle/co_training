
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

with open("../data.pkl", 'rb') as f:
    data = pickle.load(f)


X = data.drop("Malware", axis=1)
X_system_calls = X.iloc[:, 2:289]
X_permissions = X.iloc[:, 289:]
y = data['Malware']

X_train, X_test,\
    y_train, y_test = train_test_split(X_permissions, y,
                                       test_size=0.25,
                                       random_state=42)

# param_grid = {
#     'n_estimators': [500,600,700],
#     'max_features': [None],
#     'max_depth': [None],
#     'max_leaf_nodes': [None],
# }

param_grid = {
    'n_estimators': [25, 50, 100, 150, 200, 250, 300],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [None,3, 6, 9],
    'max_leaf_nodes': [None,3, 6, 9, 12, 15],
}
# Explain why max_features is none , what id does.

grid_search = GridSearchCV(RandomForestClassifier(),
                           param_grid=param_grid)
grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_)