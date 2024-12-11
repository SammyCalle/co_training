
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

with open("../data/both/X_train.pkl", 'rb') as file:
    X_train = pickle.load(file)

with open("../data/both/X_test.pkl", 'rb') as file:
    X_test = pickle.load(file)

with open("../data/both/y_train.pkl", 'rb') as file:
    y_train = pickle.load(file)

with open("../data/both/y_test.pkl", 'rb') as file:
    y_test = pickle.load(file)

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
# RandomForestClassifier(n_estimators=150)
grid_search = GridSearchCV(RandomForestClassifier(),
                           param_grid=param_grid)
grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_)