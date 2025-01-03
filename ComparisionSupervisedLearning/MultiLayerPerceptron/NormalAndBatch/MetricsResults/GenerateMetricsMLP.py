import pandas as pd
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from RFCotrainingNonStationary.SplitPreProcess import SplitViews
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import pickle

with open("../Normal/not_balanced/systemcalls/model25.pkl", 'rb') as f:
    models = pickle.load(f)

with open("../../../../DataProcessing/YearData/FullData/fulldata_year.pkl", 'rb') as f:
    data_list = pickle.load(f)


def generate_all_the_predictions_lists(data_list, models) -> []:
    prediction_list = []

    both, system_calls, permissions, y = SplitViews(data_list)

    for key in models:
        generated_list = generate_prediction_list(model=models[key],
                                                  x_test=system_calls[key])

        prediction_list.append(generated_list)

    return prediction_list, y


def generate_prediction_list(model, x_test):

    prediction_x1 = model.predict(x_test)

    prediction_x1_class = np.where(prediction_x1 >= 0.5, 1, 0)

    return np.array(prediction_x1_class)

prediction_list, y = generate_all_the_predictions_lists(
    data_list=data_list, models=models)


def generate_metrics(prediction_list, y_list):
    accuracy_list = []
    recall_list = []
    precision_list = []
    f1_score_list = []

    for prediction, labels in zip(prediction_list, y_list):
        accuracy = accuracy_score(labels, prediction)
        recall = recall_score(labels, prediction)
        precision = precision_score(labels, prediction)
        f1Score = f1_score(labels, prediction)

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

with open("../Normal/not_balanced/systemcalls/metrics25.pkl", "wb") as file:
    pickle.dump(metrics_df, file)
