from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import pickle
import sys
sys.path.append("../tools")
from feature_format import featureFormat, targetFeatureSplit

PARAM_LOGISTIC = {
    'C': [1, 3, 10, 1e3, 1e6, 1e9, 1e12],
    'fit_intercept':[True, False],
    'tol': [1e-4, 1e-6, 1e-9, 1e-12],
    'warm_start': [True, False]
}

PARAM_DECISION_TREE = {
    'criterion':['gini', 'entropy'],
    'max_depth': [3,5,9, None],
    'min_samples_split':[2,3,5],
    'min_impurity_split':[1e-7, 1e-12, 1e-5]
}

PARAM_KNEIGHBORS = {
    'n_neighbors':[3,5,7,9],
    'weights':['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

def get_best(clf, labels, features):
    '''Return the classifier with the higher score from a GridSearchCV classifier'''
    clf = clf.fit(features, labels)  # fit on entire data
    print "Best estimator found by grid search:"
    print clf.best_estimator_
    print clf.best_score_
    return clf.best_estimator_, clf.best_score_

def tune_parameters(labels, features, n_components):
    '''Tune the parameter of three classifiers.
     Parameters used are listed as constants in this file

     Returns the best classifier according to the 'scoring' metric'''

    scoring = 'f1'
    folds = 10

    pca = PCA(n_components)
    pca.fit(features)
    features_reduced = pca.fit_transform(features)

    clf1 = GridSearchCV(LogisticRegression(class_weight='balanced', solver='liblinear'), PARAM_LOGISTIC, scoring=scoring, cv=folds)
    clf2 = GridSearchCV(DecisionTreeClassifier(class_weight='balanced'), PARAM_DECISION_TREE, scoring=scoring, cv=folds)
    clf3 = GridSearchCV(KNeighborsClassifier(), PARAM_KNEIGHBORS, scoring=scoring, cv=folds)

    clfs = {}
    clfs['logistic'] = get_best(clf1, labels, features_reduced)
    clfs['decisiontree'] = get_best(clf2, labels, features_reduced)
    clfs['kneighbors'] = get_best(clf3, labels, features_reduced)

    best = 0
    best_clf = None
    for key in clfs:
        score = clfs[key][1]
        if score > best:
            best = score
            best_clf = clfs[key][0]

    return best_clf

# Main was for testing
if __name__ == "__main__":
    features_list = ['poi',
                     'salary',
                     'to_messages',
                     'deferral_payments',
                     # 'total_payments',
                     'exercised_stock_options',
                     'bonus',
                     'restricted_stock',
                     'shared_receipt_with_poi',
                     'restricted_stock_deferred',
                     # 'total_stock_value',
                     'expenses',
                     # 'loan_advances',
                     'from_messages',
                     'other',
                     # 'from_this_person_to_poi',
                     # 'director_fees',
                     'deferred_income',
                     'long_term_incentive',
                     # 'from_poi_to_this_person',
                     # 'enron',
                     # 'team',
                     # 'want',
                     # 'let',
                     # 'veri',
                     # 'issu',
                     # 'provid',
                     # 'depreci',
                     'bonus_sqrt',
                     'bonus_log',
                     'exercised_stock_options_sqrt',
                     'exercised_stock_options_log',
                     'from_poi_ratio',
                     'to_poi_ratio'
                     ]
    n_components = 11
    fn = 'my_dataset_test.pkl'
    with open(fn, "r") as data_file:
        data_dict = pickle.load(data_file)
    data = featureFormat(data_dict, features_list, sort_keys=True)
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    labels, features = targetFeatureSplit(data)

    tune_parameters(labels, features, n_components)