import pickle

import keras
from keras.src.layers import Dense
from RFCotrainingNonStationary.CotrainingPreProcess import cotraining_preparation
from CoTrainingClassifier_MLP import Non_Stationary_CoTrainingClassifier

if __name__ == '__main__':

    with open("../../DataProcessing/YearData/FullData/fulldata_year.pkl", 'rb') as f:
        data_list = pickle.load(f)

    type(data_list)
    system_calls, permissions, y = cotraining_preparation(data_list, resample=False)

    print('RandomForest Non CotrainingTuningStationary')

    clf_mlp_system_calls = keras.Sequential(
        [
            Dense(52, activation='relu', input_dim=287),
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            Dense(128, activation='relu'),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ]
    )

    clf_mlp_system_calls.compile(loss='binary_crossentropy', optimizer='adam')

    clf_mlp_permissions = keras.Sequential(
        [
            Dense(52, activation='relu', input_dim=166),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ]
    )

    clf_mlp_permissions.compile(loss='binary_crossentropy', optimizer='adam')

    rf_co_ns_clf = Non_Stationary_CoTrainingClassifier(clf=clf_mlp_system_calls, clf2=clf_mlp_permissions)

    for i in range(len(system_calls) - 1):
        rf_co_ns_clf.online_cotraining(system_calls[i], permissions[i], y[i], i)

    dict_models_X1 = rf_co_ns_clf.model_X1_dict
    dict_models_X2 = rf_co_ns_clf.model_X2_dict

    with open("YearExperimentResults/Normal/non_normal/optimized/OnlyModel/not_balanced/models_X1.pkl", "wb") as file:
        pickle.dump(dict_models_X1, file)

    with open("YearExperimentResults/Normal/non_normal/optimized/OnlyModel/not_balanced/models_X2.pkl", "wb") as file:
        pickle.dump(dict_models_X2, file)
