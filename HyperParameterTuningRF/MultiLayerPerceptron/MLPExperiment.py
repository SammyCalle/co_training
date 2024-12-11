import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from imblearn.over_sampling import SMOTE
import keras
from keras.src.layers import Dense

with open("../data/both/X_train.pkl", 'rb') as file:
    X_train = pickle.load(file)

with open("../data/both/X_test.pkl", 'rb') as file:
    X_test = pickle.load(file)

with open("../data/both/y_train.pkl", 'rb') as file:
    y_train = pickle.load(file)

with open("../data/both/y_test.pkl", 'rb') as file:
    y_test = pickle.load(file)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

model = keras.Sequential(
    [
        Dense(52, activation='relu', input_dim=453),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ]
)

model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(X_train_resampled, y_train_resampled, epochs=30)
y_pred_rand = model.predict(X_test)

y_pred = np.where(y_pred_rand >= 0.5, 1, 0)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Precision
precision = precision_score(y_test, y_pred)

# Recall
recall = recall_score(y_test, y_pred)

# F1-score
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
