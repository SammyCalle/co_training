
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import train_test_split
import pickle
from river import forest, stream

with open("../data/data.pkl", 'rb') as f:
    data = pickle.load(f)

X = data.drop("Malware", axis=1)
X_both = X.iloc[:, 2:]
X_system_calls = X.iloc[:, 2:289]
X_permissions = X.iloc[:, 289:]
y = data['Malware']

X_train, X_test,\
    y_train, y_test = train_test_split(X_permissions, y,
                                       test_size=0.25,
                                       random_state=42)

prediction = []

# model_system_calls = forest.ARFClassifier(n_models= 200)
# model_both = forest.ARFClassifier(n_models=30)
model_permissions = forest.ARFClassifier(n_models=40)

stream_data = stream.iter_pandas(X_train, y_train)

for x, y in stream_data:
    model_permissions.learn_one(x, y)

stream_test = stream.iter_pandas(X_test, y_test)
for x, y in stream_test:
    prediction.append(model_permissions.predict_one(x))


# Accuracy
accuracy = accuracy_score(y_test, prediction)

# Precision
precision = precision_score(y_test, prediction)

# Recall
recall = recall_score(y_test, prediction)

# F1-score
f1 = f1_score(y_test, prediction)


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


