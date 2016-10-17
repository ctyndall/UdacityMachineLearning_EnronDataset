#!/usr/bin/python
import numpy as np

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
#features = ['salary', 'bonus']
features = ["restricted_stock_preferred", "deferred_income"]
#features = ['from_poi_to_this_person', 'from_this_person_to_poi']
data_dict.pop('TOTAL',0)
data = featureFormat(data_dict, features)
print data

### your code below
#remove 'total' datapoint
for point in data:
    salary = point[0]
    bonus = np.sqrt(point[1])
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

