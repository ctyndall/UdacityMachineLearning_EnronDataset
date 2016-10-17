import pickle
import pandas as pd
import os
import sys
import operator
sys.path.append("../tools")
from parse_out_email_text import parseOutText

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

def get_feature_counts(data_dict):
    total_nan = 0
    for name in data_dict:
        count = 0
        for key in data_dict[name]:
            if key == 'email_address' or key == 'poi':
                continue
            if data_dict[name][key] != 'NaN':
                total_nan += 1
                count += 1
        if count < 3:
            print name, ":", count
            if data_dict[name]['poi'] == True:
                print ("POI")
    print "total NaN:", total_nan, "out of {} x {}".format(len(data_dict), len(data_dict[name]) - 2)  # ignore poi and email_address
