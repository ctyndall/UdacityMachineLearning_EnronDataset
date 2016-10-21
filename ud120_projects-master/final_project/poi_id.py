#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import tester
import my_tools
import my_gridsearchCV

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
 'salary',
 'to_messages',
 'deferral_payments',
 #'total_payments',
 'exercised_stock_options',
 'bonus',
 'restricted_stock',
 'shared_receipt_with_poi',
 #'restricted_stock_deferred',
 #'total_stock_value',
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

### Load the dictionary containing the dataset
#with open("final_project_dataset.pkl", "r") as data_file:
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
my_dataset = my_tools.remove_outliers(data_dict)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = my_tools.update_bad_finance_values(my_dataset)
my_dataset = my_tools.add_message_fractions(my_dataset)
my_dataset = my_tools.add_scaled_financial(my_dataset)

## The following were not used with the final classifier
#my_dataset = my_tools.remove_rows_limited_features(my_dataset)
#my_dataset = my_tools.load_add_word_frequencies(my_dataset)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

#Initial test
for clf in [DecisionTreeClassifier, GaussianNB, KNeighborsClassifier, LogisticRegression]:
   tester.test_classifier(clf(), my_dataset, features_list)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
clf = my_gridsearchCV.tune_parameters(my_dataset, features_list)
print clf

### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier

### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

tester.test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)