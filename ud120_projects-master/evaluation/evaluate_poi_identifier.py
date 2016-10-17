#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ['poi',
## 'salary',
# 'to_messages',
# 'deferral_payments',
## 'total_payments',
 'exercised_stock_options',
 'bonus',
 'restricted_stock',
# 'shared_receipt_with_poi',
# 'restricted_stock_deferred',
 'total_stock_value',  #has strong effect
# 'expenses',
# 'loan_advances',
# 'from_messages',
# 'other',
# 'from_this_person_to_poi',
 #'director_fees',
 'deferred_income',
 'long_term_incentive'] #,
# 'from_poi_to_this_person']

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 42) 



from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)

#clf.predict(features)

print clf.score(features_test, labels_test)

#get precision and recall score
pred = clf.predict(features_test)
from sklearn.metrics import precision_score, recall_score
print precision_score(labels_test,pred )
print recall_score(labels_test,pred)

