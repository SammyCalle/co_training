
import pandas as pd


# This initial clean removes all the data that has a bad format when it comes to the year will
def initial_clean(real_legitimate, real_malware):
    i = 1980
    year = str(i) + "-"
    removelist = real_legitimate.query('HighestModDate.str.contains(@year)', engine='python')
    removelistMalware = real_malware.query('HighestModDate.str.contains(@year)', engine='python')
    real_legitimate = real_legitimate.drop(index=removelist.index)
    real_malware = real_malware.drop(index=removelistMalware.index)

    real_legitimate['HighestModDate'] = pd.to_datetime(real_legitimate['HighestModDate'])
    real_malware['HighestModDate'] = pd.to_datetime(real_malware['HighestModDate'])

    return real_legitimate, real_malware




