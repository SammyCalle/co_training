import pickle
from RFCotrainingNonStationary.CotrainingPreProcess import cotraining_preparation
from sklearn.ensemble import RandomForestClassifier
from SemiSupervisedClassifier import SemiSupervisedClassifier


if __name__ == '__main__':

    with open("../../../DataProcessing/YearData/FullData/fulldata_year.pkl", 'rb') as f:
        data_list = pickle.load(f)

    type(data_list)
    system_calls, permissions, y = cotraining_preparation(data_list, resample=False)

    print('RandomForest Non CotrainingTuningStationary')

    model_system_calls = RandomForestClassifier(max_features=None, n_estimators=200)
    model_permissions = RandomForestClassifier(max_features='log2', n_estimators=50)

    # batch_cotraining = SemiSupervisedClassifier(clf=model_system_calls, clf2=model_permissions, p=2, n=2)
    batch_cotraining = SemiSupervisedClassifier(clf=model_permissions, clf2=model_system_calls, p=2, n=2)

    for i in range(len(system_calls)-1):

        # batch_cotraining.fit(system_calls[i], permissions[i], y[i], i)
        batch_cotraining.fit(permissions[i], system_calls[i], y[i], i)

    dict_models_X1 = batch_cotraining.model_X1_dict
    dict_models_X2 = batch_cotraining.model_X2_dict

    with open("YearExperimentResults/Normal/non_normal/optimized/OnlyModel/not_balanced/Permisions/models_X1.pkl", "wb") as file:
        pickle.dump(dict_models_X1, file)

    with open("YearExperimentResults/Normal/non_normal/optimized/OnlyModel/not_balanced/Permisions/models_X2.pkl", "wb") as file:
        pickle.dump(dict_models_X2, file)

