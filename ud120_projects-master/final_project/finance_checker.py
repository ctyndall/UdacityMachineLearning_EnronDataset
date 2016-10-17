import pickle
import pandas as pd


with open("final_project_dataset_clean.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

FINANCE_FEATURES = ['salary',
    'deferral_payments',
    'bonus',
    'expenses',
    'loan_advances',
    'other',
    'director_fees',
    'deferred_income',
    'long_term_incentive']

TOTAL  = 'total_payments'


for item in data_dict:
    total = 0
    for ff in FINANCE_FEATURES:
        if data_dict[item][ff] == 'NaN':
            continue
        total += data_dict[item][ff]
    if total != 0 and total != data_dict[item][TOTAL]:
        print total
        print item, ":ERROR"