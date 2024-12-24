import pandas as pd
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from RFCotrainingNonStationary.SplitPreProcess import SplitViews
from sklearn.metrics import accuracy_score,f1_score,recall_score, precision_score
import pickle

with open("../YearExperimentResults/non_normal/optimized/OnlyModel/not_balanced/models_X1.pkl", 'rb') as f:
    models_X1 = pickle.load(f)

with open("../YearExperimentResults/non_normal/optimized/OnlyModel/not_balanced/models_X2.pkl", 'rb') as f:
    models_X2 = pickle.load(f)

with open("../../../DataProcessing/YearData/FullData/fulldata_year.pkl", 'rb') as f:
    data_list = pickle.load(f)


def generate_all_the_predictions_lists(data_list, models_X1, models_X2) -> []:
    prediction_list = []

    both, system_calls, permissions, y = SplitViews(data_list)

    for key in models_X1:
        generated_list = generate_prediction_list(model_x1=models_X1[key], model_x2=models_X2[key],
                                                  x1_test=system_calls[key],
                                                  x2_test=permissions[key])

        prediction_list.append(generated_list)

    return prediction_list, y


def generate_prediction_list(model_x1, model_x2, x1_test, x2_test):
    prediction = []

    prediction_x1 = model_x1.predict(x1_test)
    prediction_x2 = model_x2.predict(x2_test)

    for x1, x2 in zip(prediction_x1, prediction_x2):
        prediction_value = decide_prediction(prediction_x1=x1, prediction_x2=x2)
        prediction.append(prediction_value)

    return np.array(prediction)


def decide_prediction(prediction_x1, prediction_x2):

    prediction_x1_class = np.where(prediction_x1 >= 0.5, 1, 0)
    prediction_x2_class = np.where(prediction_x2 >= 0.5, 1, 0)

    if prediction_x1_class == prediction_x2_class:
        return prediction_x1_class[0]
    else:
        sumPrediction = prediction_x1[0] + prediction_x2[0]
        pred_val = np.where(sumPrediction >= 1, 1, 0)
        return pred_val


prediction_list, y = generate_all_the_predictions_lists(
    data_list=data_list, models_X1=models_X1, models_X2=models_X2)


def generate_metrics(prediction_list, y_list):
    accuracy_list = []
    recall_list = []
    precision_list = []
    f1_score_list = []

    for prediction, labels in zip(prediction_list, y_list):

        accuracy = accuracy_score(labels,prediction)
        recall = recall_score(labels,prediction)
        precision = precision_score(labels,prediction)
        f1Score = f1_score(labels,prediction)

        accuracy_list.append(accuracy)
        recall_list.append(recall)
        precision_list.append(precision)
        f1_score_list.append(f1Score)

    return accuracy_list, recall_list, precision_list, f1_score_list


accuracy_list, recall_list, precision_list, f1_score_list = generate_metrics(prediction_list, y)


metrics_df = pd.DataFrame(
    {'Accuracy': accuracy_list, 'Recall': recall_list, 'Precision': precision_list, 'F1': f1_score_list})


latex_table = metrics_df.to_latex(
    index=True,
    caption="Normal Fit Model Performance Metrics",
    label="tab:model_performance",
    escape=False,
    column_format="|c|c|c|c|c|"
)
print(latex_table)

with open("../YearExperimentResults/non_normal/optimized/OnlyModel/not_balanced/metrics.pkl", "wb") as file:
    pickle.dump(metrics_df, file)