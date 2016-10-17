import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append("../tools")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import linear_model, decomposition, datasets

features_list = ['poi',
 'salary',
 'to_messages',
 'deferral_payments',
 #'total_payments',
 'exercised_stock_options',
 'bonus',
 'restricted_stock',
 'shared_receipt_with_poi',
 'restricted_stock_deferred',
 #'total_stock_value',  #has strong effect
 'expenses',
 #'loan_advances',
 'from_messages',
 'other',
 #'from_this_person_to_poi',
 #'director_fees',
 'deferred_income',
 'long_term_incentive' ,
 #'from_poi_to_this_person',
 'bonus_log',
 'bonus_sqrt',
 'exercised_stock_options_log',
 'exercised_stock_options_sqrt',
 'from_poi_ratio',
 'to_poi_ratio'
 # 'enron',
 # 'team',
 # 'want',
 # 'let',
 # 'veri',
 # 'issu',
 #'provid'
 # 'depreci'
]

#  load data and plt pca results
fn = 'my_dataset_test.pkl'
with open(fn, "r") as data_file:
    data_dict = pickle.load(data_file)
data = featureFormat(data_dict, features_list, sort_keys=True)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
labels, features = targetFeatureSplit(data)

pca = decomposition.PCA()

###############################################################################
# Plot the PCA spectrum
pca.fit(features)

plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')
plt.show()