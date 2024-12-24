
import pickle
from RFCotrainingNonStationary.CotrainingPreProcess import cotraining_preparation
from river import forest
from Online_SemiSupervisedClassifier_River import Online_SemiSupervisedClassifier_River

if __name__ == '__main__':

    with open("../../../DataProcessing/YearData/FullData/fulldata_year.pkl", 'rb') as f:
        data_list = pickle.load(f)

    type(data_list)
    system_calls, permissions, y = cotraining_preparation(data_list, resample=False)

    print('RandomForest Non CotrainingTuningStationary')

    model_system_calls = forest.ARFClassifier(n_models=20)

    model_permissions = forest.ARFClassifier(n_models=30)

    # rf_co_ns_clf = Online_SemiSupervisedClassifier_River(clf=model_system_calls, clf2=model_permissions, p=2, n=2)
    rf_co_ns_clf = Online_SemiSupervisedClassifier_River(clf=model_permissions, clf2=model_system_calls, p=2, n=2)

    for i in range(len(system_calls)-1):
        # rf_co_ns_clf.online_cotraining(system_calls[i], permissions[i], y[i], i)
        rf_co_ns_clf.online_cotraining(permissions[i], system_calls[i], y[i], i)

    dict_models_X1 = rf_co_ns_clf.model_X1_dict
    dict_models_X2 = rf_co_ns_clf.model_X2_dict

    with open("YearExperimentResults/non_normal/optimized/OnlyModel/not_balanced/Permisions/models_X1.pkl", "wb") as file:
        pickle.dump(dict_models_X1, file)

    with open("YearExperimentResults/non_normal/optimized/OnlyModel/not_balanced/Permisions/models_X2.pkl", "wb") as file:
        pickle.dump(dict_models_X2, file)








