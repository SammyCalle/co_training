
import pickle
from RFCotrainingNonStationary.CotrainingPreProcess import cotraining_preparation
from river import forest
import numpy as np
from Online_CoTrainingClassifier_River import Non_Stationary_CoTrainingClassifier

if __name__ == '__main__':

    with open("../../DataProcessing/YearData/FullData/fulldata_year.pkl", 'rb') as f:
        data_list = pickle.load(f)

    type(data_list)
    system_calls, permissions, y = cotraining_preparation(data_list, resample=False)

    print('RandomForest Non CotrainingTuningStationary')

    model_system_calls = forest.ARFClassifier(max_features=287,n_models=200)

    model_permissions = forest.ARFClassifier(max_features=int(np.log2(166)),n_models=50)

    rf_co_ns_clf = Non_Stationary_CoTrainingClassifier(clf=model_system_calls, clf2=model_permissions)
    for i in range(len(system_calls)-1):
        rf_co_ns_clf.online_cotraining(system_calls[i], permissions[i], y[i], i)

    dict_models_X1 = rf_co_ns_clf.model_X1_dict
    dict_models_X2 = rf_co_ns_clf.model_X2_dict

    with open("YearExperimentResults/non_normal/optimized/OnlyModel/balanced/models_X1.pkl", "wb") as file:
        pickle.dump(dict_models_X1, file)

    with open("YearExperimentResults/non_normal/optimized/OnlyModel/balanced/models_X2.pkl", "wb") as file:
        pickle.dump(dict_models_X2, file)








