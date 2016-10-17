#!/usr/bin/python

import pickle
import pandas as pd


with open("final_project_dataset_clean.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

df = pd.DataFrame.from_dict(data_dict).T
#df.to_csv('data.csv')
df = df.reset_index()

del df['index']
del df['poi']
del df['email_address']
del df['loan_advances']
del df['director_fees']

print df
print df.corr()