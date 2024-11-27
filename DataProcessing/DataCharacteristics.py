import pickle
import pandas as pd

with open("YearData/FullData/fulldata_year.pkl", 'rb') as f:
    data_list = pickle.load(f)

summary_df = pd.DataFrame(columns=['Dataset', 'Count_0', 'Count_1'])

summary_data = []
for i in range(len(data_list)):

    count_0 = data_list[i]['Malware'].value_counts()[0]
    count_1 = data_list[i]['Malware'].value_counts()[1]
    summary_data.append({'Dataset': f'Dataset {i+1}', 'Count_0': count_0, 'Count_1': count_1})

summary_df = pd.DataFrame.from_records(summary_data)


latex_table = summary_df.to_latex(
    index=True,
    caption="Class Balance",
    label="tab:class_balance",
    escape=False,
    column_format="|c|c|c|c|c|"
)
print(latex_table)
