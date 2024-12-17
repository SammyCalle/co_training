import pandas as pd
import random
from RFCotrainingNonStationary.SplitPreProcess import SplitViews
from river import stream, metrics, forest
import pickle

with open("../balanced/both/model25.pkl", 'rb') as f:
    models = pickle.load(f)

with open("../../../../DataProcessing/YearData/FullData/fulldata_year.pkl", 'rb') as f:
    data_list = pickle.load(f)


def generate_all_the_predictions_lists(data_list, models) -> []:
    prediction_list = []

    both, system_calls, permissions, y = SplitViews(data_list)

    for key in models:
        generated_list = generate_prediction_list(models=models[key],
                                                  x_test=both[key], y=y[key])

        prediction_list.append(generated_list)

    return prediction_list, y


def generate_prediction_list(models, x_test, y) -> []:
    prediction = []

    prediction_stream_x1 = stream.iter_pandas(x_test, y)
    for x, label in prediction_stream_x1:
        prediction.append(models.predict_one(x))

    return prediction

prediction_list, y = generate_all_the_predictions_lists(
    data_list=data_list, models=models)


def generate_metrics(prediction_list, y_list):
    accuracy_list = []
    recall_list = []
    precision_list = []
    f1_score_list = []

    for prediction, labels in zip(prediction_list, y_list):
        accuracy = metrics.Accuracy()
        recall = metrics.Recall()
        precision = metrics.Precision()
        f1_score = metrics.F1()

        for yt, yp in zip(prediction, labels):
            accuracy.update(yt, yp)
            recall.update(yt, yp)
            precision.update(yt, yp)
            f1_score.update(yt, yp)

        accuracy_list.append(accuracy.get())
        recall_list.append(recall.get())
        precision_list.append(precision.get())
        f1_score_list.append(f1_score.get())

    return accuracy_list, recall_list, precision_list, f1_score_list


accuracy_list, recall_list, precision_list, f1_score_list = generate_metrics(prediction_list, y)

metrics_df = pd.DataFrame(
    {'Accuracy': accuracy_list, 'Recall': recall_list, 'Precision': precision_list, 'F1': f1_score_list})

latex_table = metrics_df.to_latex(
    index=True,
    caption="Model Performance Metrics",
    label="tab:model_performance",
    escape=False,
    column_format="|c|c|c|c|c|"
)
print(latex_table)

with open("../balanced/both/metrics25.pkl", "wb") as file:
    pickle.dump(metrics_df, file)
