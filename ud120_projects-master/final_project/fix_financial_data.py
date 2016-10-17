import pickle
import pandas as pd

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

def check_finance_sums(data_dict):
    FINANCE_FEATURES = ['salary',
                        'deferral_payments',
                        'bonus',
                        'expenses',
                        'loan_advances',
                        'other',
                        'director_fees',
                        'deferred_income',
                        'long_term_incentive']

    PAYMENTS_TOTAL  = 'total_payments'

    STOCK_FEATURES = ['exercised_stock_options',
                      'restricted_stock',
                      'restricted_stock_deferred']
    STOCK_TOTAL = 'total_stock_value'

    for item in data_dict:
        total = 0
        for ff in FINANCE_FEATURES:
            if data_dict[item][ff] == 'NaN':
                continue
            total += data_dict[item][ff]
        if total != 0 and total != data_dict[item][PAYMENTS_TOTAL]:
            print total
            print item, ":PAYMENT SUM ERROR"

        total = 0
        for sf in STOCK_FEATURES:
            if data_dict[item][sf] == 'NaN':
                continue
            total += data_dict[item][sf]
        if total != 0 and total != data_dict[item][STOCK_TOTAL]:
            print total
            print item, ":STOCK SUM ERROR"

def update_values(data_dict):
    data_dict['BELFER ROBERT'].update({'salary': 'NaN',
                                  'bonus': 'NaN',
                                  'long_term_incentive': 'NaN',
                                  'deferred_income': -102500,
                                  'deferral_payments': 'NaN',
                                  'loan_advances': 'NaN',
                                  'other': 'NaN',
                                  'expenses': 3285,
                                  'director_fees': 102500,
                                  'total_payments': 3285,
                                  'exercised_stock_options': 'NaN',
                                  'restricted_stock': 44093,
                                  'restricted_stock_deferred': -44093,
                                  'total_stock_value': 'NaN'})

    data_dict['BHATNAGAR SANJAY'].update({'salary': 'NaN',
                                  'bonus': 'NaN',
                                  'long_term_incentive': 'NaN',
                                  'deferred_income': 'NaN',
                                  'deferral_payments': 'NaN',
                                  'loan_advances': 'NaN',
                                  'other': 'NaN',
                                  'expenses': 137864,
                                  'director_fees': 'NaN',
                                  'total_payments': 137864,
                                  'exercised_stock_options': 15456290,
                                  'restricted_stock': 2604490,
                                  'restricted_stock_deferred': -2604490,
                                  'total_stock_value': 15456290})

    return data_dict

# check_finance_sums(data_dict)
# print "checking updated data..."
# check_finance_sums(update_values(data_dict))