
import pickle
from RFCotrainingNonStationary.CotrainingPreProcess import cotraining_preparation
from river import forest
from Online_CoTrainingClassifier_River import Non_Stationary_CoTrainingClassifier
from imblearn.over_sampling import SMOTE

if __name__ == '__main__':

    with open("../../DataProcessing/YearData/FullData_MinMax/fulldata_year.pkl", 'rb') as f:
        data_list = pickle.load(f)

    type(data_list)
    system_calls, permissions, y = cotraining_preparation(data_list, resample=True)

    print('RandomForest Non RFCotrainingStationary')

    rf_co_ns_clf = Non_Stationary_CoTrainingClassifier(forest.ARFClassifier())

    for i in range(len(system_calls)-1):
        rf_co_ns_clf.online_cotraining(system_calls[i], permissions[i], y[i], i)

    dict_models_X1 = rf_co_ns_clf.model_X1_dict
    dict_models_X2 = rf_co_ns_clf.model_X2_dict

    with open("YearExperimentResults/min_max_scaler/balanced/models_X1.pkl", "wb") as file:
        pickle.dump(dict_models_X1, file)

    with open("YearExperimentResults/min_max_scaler/balanced/models_X2.pkl", "wb") as file:
        pickle.dump(dict_models_X2, file)








