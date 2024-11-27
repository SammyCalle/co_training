# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from CoTrainingClassifier import CoTrainingClassifier

if __name__ == '__main__':
    with open("../data/permissions/X_train.pkl", 'rb') as f:
        permissions = pickle.load(f)

    with open("../data/syscalls/X_train.pkl", 'rb') as f:
        syscalls = pickle.load(f)

    with open("../data/permissions/y_train.pkl", 'rb') as f:
        labels = pickle.load(f)

    list_data_permissions = permissions[:12]
    list_data_syscalls = syscalls[:12]
    list_y = labels[:12]

    data_permissions = np.vstack(list_data_permissions)
    data_syscalls = np.vstack(list_data_syscalls)
    y = np.concatenate(list_y)
    print(f"La cantidad de 0 : {np.count_nonzero(y == 0)}")
    print(f"La cantidad de 1 : {np.count_nonzero(y == 1)}")
    print(f"Shape de data_syscalls previo a distribucion :{data_syscalls.shape}")
    print(f"Shape de Y previo a distribucion :{y.shape}")

    N_SAMPLES = len(data_syscalls)

    y[:N_SAMPLES // 2] = -1

    data_permissions_test = data_permissions[-N_SAMPLES // 4:]
    data_syscalls_test = data_syscalls[-N_SAMPLES // 4:]
    y_test = y[-N_SAMPLES // 4:]

    # data_permissions_labeled = data_permissions[:N_SAMPLES//4]
    # data_syscalls_labeled = data_syscalls[:N_SAMPLES//4]
    # y_labeled = y[:N_SAMPLES // 4]

    data_permissions_labeled = data_permissions[N_SAMPLES // 2:-N_SAMPLES // 4]
    data_syscalls_labeled = data_syscalls[N_SAMPLES // 2:-N_SAMPLES // 4]
    y_labeled = y[N_SAMPLES // 2:-N_SAMPLES // 4]

    # data_permissions_unlabeled = data_permissions[N_SAMPLES//4:-N_SAMPLES//4]
    # data_syscalls_unlabeled = data_syscalls[N_SAMPLES//4:-N_SAMPLES//4]
    # y_unlabeled = y[N_SAMPLES // 4:-N_SAMPLES // 4]

    X_labeled = np.hstack((data_permissions_labeled, data_syscalls_labeled))
    X_test = np.hstack((data_permissions_test, data_syscalls_test))

    print(data_permissions_labeled.shape)
    print(data_syscalls_labeled.shape)
    print(f"El shape de X_labeled es :{X_labeled.shape}")
    print(f"El shape de y_labeled es :{y_labeled.shape}")
    print(f"La cantidad de 0 : {np.count_nonzero(y_labeled == 0)}")
    print(f"La cantidad de 1 : {np.count_nonzero(y_labeled == 1)}")

    print('RandomForestClassifier')
    base_rf = RandomForestClassifier()
    base_rf.fit(X_labeled, y_labeled)
    y_pred = base_rf.predict(X_test)
    print(classification_report(y_test, y_pred))

    print('RandomForestClassifier CoTraining')
    rf_co_clf = CoTrainingClassifier(RandomForestClassifier())
    rf_co_clf.fit(X1=data_permissions, X2=data_syscalls, y=y)
    y_pred = rf_co_clf.predict(data_permissions_test,data_syscalls_test)
    print(classification_report(y_test, y_pred))

    # print('AdaBoost CoTraining')
