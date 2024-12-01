import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import keras
from keras.src.layers import Dense

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

model.fit(X_train, y_train, epochs=10)
y_pred_rand = model.predict(X_test)


y_pred_rand_flattened = y_pred_rand.reshape(-1)
# Create a DataFrame with indices and predictions
df_results = pd.DataFrame({'index': X_test.index, 'predicted_probability': y_pred_rand_flattened})

# Sort by predicted probability in descending order
df_results = df_results.sort_values(by='predicted_probability', ascending=False)

# Assuming df_results is your DataFrame with 'predicted_probability' column
n = 5  # Number of largest and smallest values to select

# Get the indices of the n largest values
largest_indices = df_results.nlargest(n, 'predicted_probability').index

# Get the indices of the n smallest values
smallest_indices = df_results.nsmallest(n, 'predicted_probability').index

# Extract the corresponding rows from the original DataFrame (X_test)
largest_rows = X_test.loc[largest_indices]
smallest_rows = X_test.loc[smallest_indices]

y_pred = np.where(y_pred_rand >= 0.5, 1, 0)
print(classification_report(y_pred, y_test))
