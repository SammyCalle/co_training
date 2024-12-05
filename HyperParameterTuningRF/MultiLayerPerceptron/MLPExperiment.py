import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import keras
from keras.src.layers import Dense


with open("../data/permissions/X_train_permissions.pkl", 'rb') as file:
    X_train_permissions = pickle.load(file)

with open("../data/permissions/X_test_permissions.pkl", 'rb') as file:
    X_test_permissions = pickle.load(file)

with open("../data/permissions/y_train_permissions.pkl", 'rb') as file:
    y_train_permissions = pickle.load(file)

with open("../data/permissions/y_test_permissions.pkl", 'rb') as file:
    y_test_permissions = pickle.load(file)


model = keras.Sequential(
        [
            Dense(52, activation='relu', input_dim=165),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ]
    )

model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(X_train_permissions, y_train_permissions, epochs=10)
y_pred_rand = model.predict(X_test_permissions)


y_pred_rand_flattened = y_pred_rand.reshape(-1)
# # Create a DataFrame with indices and predictions
# df_results = pd.DataFrame({'index': X_test_permissions.index, 'predicted_probability': y_pred_rand_flattened})
#
# # Sort by predicted probability in descending order
# df_results = df_results.sort_values(by='predicted_probability', ascending=False)
#
# # Assuming df_results is your DataFrame with 'predicted_probability' column
n = 5  # Number of largest and smallest values to select
#
# # Get the indices of the n largest values
# largest_indices = df_results.nlargest(n, 'predicted_probability').index
#
# # Get the indices of the n smallest values
# smallest_indices = df_results.nsmallest(n, 'predicted_probability').index
#
# # Extract the corresponding rows from the original DataFrame (X_test)
# largest_rows = X_test_permissions.loc[largest_indices.values]
# smallest_rows = X_test_permissions.loc[smallest_indices.values]

# Get indices of sorted probabilities
sorted_indices = np.argsort(y_pred_rand_flattened)[::-1]  # Descending order
top_n_indices = sorted_indices[:n]
bottom_n_indices = sorted_indices[-n:]

# Extract top and bottom n rows
top_n_rows = X_test_permissions.iloc[top_n_indices]
bottom_n_rows = X_test_permissions.iloc[bottom_n_indices]

y_pred = np.where(y_pred_rand >= 0.5, 1, 0)

# Accuracy
accuracy = accuracy_score(y_test_permissions, y_pred)

# Precision
precision = precision_score(y_test_permissions, y_pred)

# Recall
recall = recall_score(y_test_permissions, y_pred)

# F1-score
f1 = f1_score(y_test_permissions, y_pred)


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
