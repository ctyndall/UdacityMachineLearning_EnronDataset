import pickle
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../tools")
from feature_format import featureFormat, targetFeatureSplit

with open("my_dataset2.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

df = pd.DataFrame.from_dict(data_dict).T
df = df.replace('NaN', np.nan)
feature = 'bonus'
features = ['poi', 'concern']
#print df[['poi','bonus']]
#print df['bonus'].asnumeric()
print df[feature]

def plot_hist(df, feature):
    df.plot(x=features[0], y=features[1], kind='scatter')
    plt.show()
plot_hist(df, feature)