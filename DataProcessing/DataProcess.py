from DataCleanUtil import initial_clean
from DateSplitterUtil import date_constraining
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler

mode_SystemCalls = 0
mode_Permissions = 1
mode_Both = 2
mode_labels = 3

real_legitimate = pd.read_csv('real_legitimate_v1.csv')
real_malware = pd.read_csv('real_malware_v1.csv')

real_legitimate, real_malware = initial_clean(real_legitimate=real_legitimate, real_malware=real_malware)


# dfstreeMonth = date_constraining(initial_date="2012-01-01", final_date="2018-12-31"
#                                  , splitter='12MS',real_legitimate=real_legitimate
#                                  ,real_malware=real_malware, mode=mode_Both)

dfsNOSplit = date_constraining(initial_date="2012-01-01", final_date="2018-12-31"
                                ,real_legitimate=real_legitimate
                                 ,real_malware=real_malware, mode=mode_Both)

# for eachdataset in dfstreeMonth:
#     scaler = StandardScaler()
#     eachdataset = scaler.fit_transform(eachdataset)

with open("../HyperParameterTuningRF/data.pkl", "wb") as f:
    pickle.dump(dfsNOSplit, f)