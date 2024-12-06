
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

with open("../data/permissions/X_train_permissions.pkl", 'rb') as file:
    X_train_permissions = pickle.load(file)

with open("../data/permissions/X_test_permissions.pkl", 'rb') as file:
    X_test_permissions = pickle.load(file)

with open("../data/permissions/y_train_permissions.pkl", 'rb') as file:
    y_train_permissions = pickle.load(file)

with open("../data/permissions/y_test_permissions.pkl", 'rb') as file:
    y_test_permissions = pickle.load(file)

# param_grid = {
#     'n_estimators': [500,600,700],
#     'max_features': [None],
#     'max_depth': [None],
#     'max_leaf_nodes': [None],
# }

param_grid = {
    'n_estimators': [25, 50, 100, 150, 200, 250, 300, 600],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [None,3, 6, 9],
    'max_leaf_nodes': [None,3, 6, 9, 12, 15],
}

grid_search = GridSearchCV(RandomForestClassifier(),
                           param_grid=param_grid)
grid_search.fit(X_train_permissions, y_train_permissions)
print(grid_search.best_estimator_)