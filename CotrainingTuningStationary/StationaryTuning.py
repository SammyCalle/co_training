
import pickle
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from CoTrainingClassifier import CoTrainingClassifier

if __name__ == '__main__':
    with open("data/data.pkl", 'rb') as f:
        data = pickle.load(f)

    X = data.drop("Malware", axis=1)
    y = data['Malware']
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, \
        y_train, y_test = train_test_split(X_resampled, y_resampled,
                                           test_size=0.25,
                                           random_state=42)

    num_rows = len(y_train)
    # Calculate the number of samples to change
    num_to_change = int(num_rows * 0.25)

    y_train.iloc[-num_to_change:] = -1

    X_train_system_calls = X_train.iloc[:, 2:289]
    X_train_permissions = X_train.iloc[:, 289:]

    X_test_system_calls = X_test.iloc[:, 2:289]
    X_test_permissions = X_test.iloc[:, 289:]

    model = RandomForestClassifier()
    # model_system_calls = RandomForestClassifier(max_features=None, n_estimators=200)
    # model_permissions = RandomForestClassifier(max_features='log2', n_estimators=50)

    print('RandomForestClassifier CoTraining')
    rf_co_clf = CoTrainingClassifier(clf=model, k=10)
    rf_co_clf.fit(X1=X_train_system_calls, X2=X_train_permissions, y=y_train)
    y_pred = rf_co_clf.predict(X_test_system_calls, X_test_permissions)

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

