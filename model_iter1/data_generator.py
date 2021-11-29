import pandas as pd
import numpy as np

df = pd.read_csv ('data_summary_final_summary.csv')
df
#df.loc[df['Native_Chip_Name'] = some_value]
df[df['Native_Chip_Name'].str.contains("s2019354")]['EpochTime'].iloc[0]
df.rename(columns={'Unnamed: 0':'ID'}, inplace=True)
df.rename(columns={'05min_Lightning_Count':'label'}, inplace=True)

train_set, test_set= np.split(df, [int(.65 *len(df))])

pd.Series(df.label,index=df.ID).to_dict()