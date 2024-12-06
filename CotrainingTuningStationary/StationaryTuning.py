
import pickle
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from CoTrainingClassifier import CoTrainingClassifier

if __name__ == '__main__':
    with open("data/data.pkl", 'rb') as f:
        data = pickle.load(f)

    num_rows = len(data)
    # Calculate the number of samples to change
    num_to_change = int(num_rows * 0.25)

    data.loc[num_to_change:, 'Malware'] = -1

    X = data.drop("Malware", axis=1)
    y = data['Malware']

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, \
        y_train, y_test = train_test_split(X_resampled, y_resampled,
                                           test_size=0.25,
                                           random_state=42)

    X_train_system_calls = X_train.iloc[:, 2:289]
    X_train_permissions = X_train.iloc[:, 289:]

    X_test_system_calls = X_test.iloc[:, 2:289]
    X_test_permissions = X_test.iloc[:, 289:]

    model_system_calls = RandomForestClassifier(max_features=None, n_estimators=200)
    model_permissions = RandomForestClassifier(max_features='log2', n_estimators=50)

    print('RandomForestClassifier CoTraining')
    rf_co_clf = CoTrainingClassifier(clf=model_system_calls,clf2=model_permissions)
    rf_co_clf.fit(X1=X_train_system_calls, X2=X_train_permissions, y=y_train)
    y_pred = rf_co_clf.predict(X_test_system_calls,X_test_permissions)
    print(classification_report(y_test, y_pred))

    # print('AdaBoost CoTraining')
