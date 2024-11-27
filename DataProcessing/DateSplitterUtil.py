
import pandas as pd

# This function returns a dataframe or list of dataframes depending if the slipter options
# is given , this dataframes will be created from the big dataframe and will be constrained
# by the date of the "HighestModDate".
def date_constraining(initial_date, final_date, real_legitimate, real_malware, mode, splitter=False):
    # Some other columns that will be droped from the dataset because are not needed for the classification
    drops = ['Package', 'MalFamily', 'nr_permissions', 'normal', 'dangerous', 'signature', 'custom_yes', 'nr_custom'
        , 'total_perm', 'sha256', 'CFileSize', 'UFileSize', 'EarliestModDate'
        , 'Detection_Ratio', 'nr_syscalls', 'Scanners', 'FilesInsideAPK', 'TimesSubmitted', 'NrContactedIps']
    listStr = ['Activities', 'NrIntServices', 'NrIntServicesActions', 'NrIntActivities', 'NrIntActivitiesActions',
               'NrIntReceivers'
        , 'NrIntReceiversActions', 'TotalIntentFilters', 'NrServices']

    # This create two list one with all the column names that belongs to the SysCalls and
    # the other one for the permissions , this list has been created following the description
    # of the dataset in the paper KronoDroid
    sys_calls = []
    permission = []
    counter = 1
    for i in real_legitimate.columns:
        if (counter > 2 and counter < 291):
            sys_calls.append(i)
        if (counter > 291 and counter < 458):
            permission.append(i)
        counter = counter + 1

    # mode will determinate if the output dataframe will use :
    # 0 = Only SystemCallls
    # 1 = Only Permissions
    # 2 = Both
    real_legitimate = real_legitimate.loc[
        (real_legitimate['HighestModDate'] >= initial_date) & (real_legitimate['HighestModDate'] <= final_date)]
    real_malware = real_malware.loc[
        (real_malware['HighestModDate'] >= initial_date) & (real_malware['HighestModDate'] <= final_date)]
    data = pd.concat([real_legitimate, real_malware], ignore_index=True)

    data = data.drop(drops, axis=1)
    data = data.drop(listStr, axis=1)
    if mode == 0:
        data = data.drop(permission,axis= 1)
        data = data.drop('Malware', axis=1)
    elif mode == 1:
        data = data.drop(sys_calls,axis= 1)
        data = data.drop('Malware', axis=1)
    elif mode == 2:
        data = data
    elif mode == 3:
        data = data[['Malware', 'HighestModDate']]

    data = data.sample(frac=1)
    if not splitter:
        return data.drop(columns='HighestModDate')
    else:
        treeMonth = data.groupby(pd.Grouper(key='HighestModDate', freq=splitter))
        dfstreeMonth = [group.drop(columns='HighestModDate').reset_index(drop=True) for _, group in treeMonth]
        # dfstreeMonth = [group.drop(columns='HighestModDate') for _, group in treeMonth]
        return dfstreeMonth
