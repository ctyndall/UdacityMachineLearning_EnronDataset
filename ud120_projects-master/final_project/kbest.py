import sys
sys.path.append("../tools")
from feature_format import featureFormat, targetFeatureSplit
from my_tools import *

features_list = ['poi',
 'salary',
 'to_messages',
 'deferral_payments',
 'total_payments',
 'exercised_stock_options',
 'bonus',
 'restricted_stock',
 'shared_receipt_with_poi',
 'restricted_stock_deferred',
 'total_stock_value',  #has strong effect
 'expenses',
 'loan_advances',
 'from_messages',
 'other',
 'from_this_person_to_poi',
 'director_fees',
 'deferred_income',
 'long_term_incentive' ,
 'from_poi_to_this_person',
 'bonus_log',
 'bonus_sqrt',
 'exercised_stock_options_log',
 'exercised_stock_options_sqrt',
 'from_poi_ratio',
 'to_poi_ratio',
  'enron',
  'team',
  'want',
  'let',
  'veri',
  'issu',
  'provid',
  'depreci'
]

def get_kbest(data_dict, features_list):
    from sklearn.feature_selection import SelectKBest
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)
    kbest = SelectKBest()
    kbest.fit(features, labels)
    print kbest.get_support()
    print kbest.get_params()
    f_scores = sorted(kbest.scores_)
    for score, feature in sorted(zip(kbest.scores_, features_list[1:]), key=lambda x:x[0], reverse = True):
        print score, feature
    return f_scores

fn = "my_dataset_test.pkl"
data_dict = load_dict(fn)
#data_dict = remove_outliers(data_dict)
data_dict = add_scaled_financial(data_dict)
data_dict = add_message_fractions(data_dict)
#data_dict = add_word_ratios(data_dict, get_emails(data_dict))

f_scores = get_kbest(data_dict, features_list)

import matplotlib.pyplot as plt
plt.plot(f_scores)
plt.show()