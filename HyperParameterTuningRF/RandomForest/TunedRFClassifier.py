
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier

with open("../data/data.pkl", 'rb') as f:
    data = pickle.load(f)

X = data.drop("Malware", axis=1)
X_system_calls = X.iloc[:, 2:289]
X_permissions = X.iloc[:, 289:]
y = data['Malware']

X_train, X_test,\
    y_train, y_test = train_test_split(X_system_calls, y,
                                       test_size=0.25,
                                       random_state=42)


# Experiments with system calls
# RandomForestClassifier(max_depth=9, max_features=None, n_estimators=150)
# RandomForestClassifier(max_depth=9, max_features=None, max_leaf_nodes=60,n_estimators=150)
# RandomForestClassifier(max_depth=15, max_features=None, n_estimators=300)
# RandomForestClassifier(max_features=None, n_estimators=300)
# RandomForestClassifier(max_features=None, n_estimators=600)

model = RandomForestClassifier(max_features=None, n_estimators=200)

# Experiment with permissions
# RandomForestClassifier(n_estimators=50)
# model = RandomForestClassifier(max_features='log2', n_estimators=50)

model.fit(X_train, y_train)
y_pred_rand = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred_rand)

# Precision
precision = precision_score(y_test, y_pred_rand)

# Recall
recall = recall_score(y_test, y_pred_rand)

# F1-score
f1 = f1_score(y_test, y_pred_rand)


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


