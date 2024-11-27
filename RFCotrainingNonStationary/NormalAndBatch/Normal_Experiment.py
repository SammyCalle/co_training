import pickle
from RFCotrainingNonStationary.CotrainingPreProcess import cotraining_preparation
from sklearn.ensemble import RandomForestClassifier
from CoTrainingClassifier import CoTrainingClassifier


if __name__ == '__main__':

    with open("../../DataProcessing/YearData/FullData_Standard_Scaler/fulldata_year.pkl", 'rb') as f:
        data_list = pickle.load(f)

    type(data_list)
    system_calls, permissions, y = cotraining_preparation(data_list, resample=True)

    print('RandomForest Non RFCotrainingStationary')

    batch_cotraining = CoTrainingClassifier(RandomForestClassifier())

    for i in range(len(system_calls)-1):

        batch_cotraining.fit(system_calls[i], permissions[i], y[i], i)

    dict_models_X1 = batch_cotraining.model_X1_dict
    dict_models_X2 = batch_cotraining.model_X2_dict

    with open("YearExperimentResults/Normal/standard_scaler/models_X1.pkl", "wb") as file:
        pickle.dump(dict_models_X1, file)

    with open("YearExperimentResults/Normal/standard_scaler/models_X2.pkl", "wb") as file:
        pickle.dump(dict_models_X2, file)

