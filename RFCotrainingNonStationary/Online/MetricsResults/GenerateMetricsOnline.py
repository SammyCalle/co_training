import pandas as pd
import random
from RFCotrainingNonStationary.SplitPreProcess import SplitViews
from river import stream, metrics, forest
import pickle

with open("../YearExperimentResults/non_normal/optimized/OnlyModel/balanced/models_X1.pkl", 'rb') as f:
    models_X1 = pickle.load(f)

with open("../YearExperimentResults/non_normal/optimized/OnlyModel/balanced/models_X2.pkl", 'rb') as f:
    models_X2 = pickle.load(f)

with open("../../../DataProcessing/YearData/FullData_MinMax/fulldata_year.pkl", 'rb') as f:
    data_list = pickle.load(f)


def generate_all_the_predictions_lists(data_list, models_X1, models_X2) -> []:
    prediction_list = []

    system_calls, permissions, y = SplitViews(data_list)

    for key in models_X1:
        generated_list = generate_prediction_list(model_x1=models_X1[key], model_x2=models_X2[key],
                                                  x1_test=system_calls[key],
                                                  x2_test=permissions[key], y=y[key])

        prediction_list.append(generated_list)

    return prediction_list, y


def generate_prediction_list(model_x1, model_x2, x1_test, x2_test, y) -> []:
    prediction_x1 = []
    prediction_x2 = []
    prediction = []

    prediction_stream_x1 = stream.iter_pandas(x1_test, y)
    for x, label in prediction_stream_x1:
        prediction_x1.append(model_x1.predict_one(x))

    prediction_stream_x2 = stream.iter_pandas(x2_test, y)
    for x, label in prediction_stream_x2:
        prediction_x2.append(model_x2.predict_one(x))

    for i, (y1, y2) in enumerate(zip(prediction_x1, prediction_x2)):
        if y1 == y2:
            prediction.append(int(y1))
        else:
            y1_probs = model_x1.predict_proba_one(x1_test.iloc[i].to_dict())
            y2_probs = model_x2.predict_proba_one(x2_test.iloc[i].to_dict())
            sum_y_probs = [prob1 + prob2 for (prob1, prob2) in zip(y1_probs, y2_probs)]
            max_sum_prob = max(sum_y_probs)
            prediction.append(int(sum_y_probs.index(max_sum_prob)))

    # for x1, x2 in zip(prediction_x1, prediction_x2):
    #     prediction.append(decide_prediction(prediction_x1=x1, prediction_x2=x2))

    return prediction


def decide_prediction(prediction_x1, prediction_x2) -> int:
    if prediction_x1 == prediction_x2:
        return prediction_x1
    else:
        return random.randint(0, 1)


prediction_list, y = generate_all_the_predictions_lists(
    data_list=data_list, models_X1=models_X1, models_X2=models_X2)


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

with open("../YearExperimentResults/non_normal/optimized/OnlyModel/balanced/metrics.pkl", "wb") as file:
    pickle.dump(metrics_df, file)
