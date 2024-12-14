
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import train_test_split
import pickle
from skmultiflow.meta import AdaptiveRandomForestClassifier

with open("../data/data.pkl", 'rb') as f:
    data = pickle.load(f)

X = data.drop("Malware", axis=1)
X_both = X.iloc[:, 2:]
X_system_calls = X.iloc[:, 2:289]
X_permissions = X.iloc[:, 289:]
y = data['Malware']

X_train, X_test,\
    y_train, y_test = train_test_split(X_system_calls.to_numpy(), y.to_numpy(),
                                       test_size=0.25,
                                       random_state=42)


model_system_calls = AdaptiveRandomForestClassifier(max_features=287,n_estimators=10)

model_system_calls.partial_fit(X_train, y_train)

y_pred = model_system_calls.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, X_test)

# Precision
precision = precision_score(y_test, X_test)

# Recall
recall = recall_score(y_test, X_test)

# F1-score
f1 = f1_score(y_test, X_test)


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


