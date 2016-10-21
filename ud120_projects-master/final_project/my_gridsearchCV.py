from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import pickle
import sys
sys.path.append("../tools")
from feature_format import featureFormat, targetFeatureSplit

PARAM_LOGISTIC_GRID = {
    'clf__C' : [1, 5, 10, 1e3, 1e6],
    #'clf__fit_intercept' : [True, False], #, False]
    'clf__intercept_scaling' : [0.1, 1, 10],
    'clf__tol' : [1e-4, 1e-6, 1e-9]
}

PARAM_DECISION_TREE_GRID = {
    'clf__criterion':['gini', 'entropy'],
    'clf__max_depth': [3,5,9, None],
    'clf__min_samples_split':[2,3,5],
    'clf__min_impurity_split': [1e-5, 1e-7, 1e-12]
}

PARAM_KNEIGHBORS_GRID = {
    'clf__n_neighbors':[3,5,7,9],
    'clf__weights':['uniform', 'distance'],
    'clf__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

PARAM_PCA = {
    'pca__n_components':[3, 5, 7, 9, 11, 13] #, 11, 13]#11,13,15] #n_components
}

PARAM_KBEST  = {
    'kb__k': [13, 15, 17, 'all'] #k 23, 27, 29, 31
}

def get_best(clf, labels, features, features_list):
    '''Return the classifier with the higher score from a GridSearchCV classifier'''
    clf.fit(features, labels)  # fit on entire data
    print "Best estimator found by grid search:"
    print clf.best_estimator_

    mask = clf.best_estimator_.named_steps['kb'].get_support()
    best_features = [features_list[1:][i] for i, bool in enumerate(mask) if bool]
    removed_features = [features_list[1:][i] for i, bool in enumerate(mask) if not bool]

    print "Top {}/{} Features Kept: ".format(len(best_features), len(features_list[1:])), best_features
    print "Features Scores: ", clf.best_estimator_.named_steps['kb'].scores_
    print "Best estimator found by grid search:" ,  str(clf.best_estimator_)
    print "Number of PCA components used: ", clf.best_estimator_.named_steps['pca'].n_components_
    print "Best classifer: ", clf.best_estimator_.named_steps['clf']
    print "Score: ", clf.best_score_
    print "Params:",  clf.best_params_

    with open('results.txt', 'a') as output:
        output.write("Best parameters: " +  str(clf.best_params_) + "\n")
        output.write("Top {}/{} Features Kept: ".format(len(best_features), len(features_list[1:]))
                     + str(best_features) + "\n") #str(clf.best_estimator_.named_steps['kb'].get_support()) + "\n")
        output.write("Features {}/{} Removed: ".format(len(removed_features), len(features_list[1:]))
                     + str(removed_features) + "\n")
        output.write("Features Scores: " + str(clf.best_estimator_.named_steps['kb'].scores_) + "\n")
        # output.write("Features parameters: " + str(clf.best_estimator_.named_steps['kb'].get_params()) + "\n")
        output.write("Number of PCA components used: " + str(clf.best_estimator_.named_steps['pca'].n_components_) + "\n")
        output.write("Best classifier: " + str(clf.best_estimator_.named_steps['clf']) + "\n")
        output.write("F1 Score: " + str(clf.best_score_) + "\n")
        output.write("---------------------------------------------------\n\n")

    return clf.best_estimator_, clf.best_score_

def tune_parameters(data_set, features_list):
    '''Tune the parameter of three classifiers.
     Parameters used are listed as constants in this file

     Returns the best classifier according to the 'scoring' metric'''

    data = featureFormat(data_set, features_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)
    # cv = StratifiedShuffleSplit(labels, n_iter=1000, random_state = 43)
    scoring = 'f1'
    folds = 10
    # pca = PCA(n_components)
    # pca.fit(features)
    # features_reduced = pca.fit_transform(features)
    classifiers = [LogisticRegression(class_weight='balanced')] #,
                   #DecisionTreeClassifier(class_weight='balanced')] #,
                   #KNeighborsClassifier()]
    clf_params = [PARAM_LOGISTIC_GRID] #,
                 #PARAM_DECISION_TREE_GRID,
                 #PARAM_KNEIGHBORS_GRID]

    best_clfs = []
    for classifier, param in zip(classifiers, clf_params):
        pipeline = Pipeline(steps=[('scaler', MinMaxScaler()), ('kb', SelectKBest()), ('pca', PCA()), ('clf', classifier)])
        params = {}
        params.update(PARAM_KBEST)
        params.update(PARAM_PCA)
        params.update(param)
        clf = GridSearchCV(pipeline,
                            params,
                            scoring=scoring,
                            cv=folds)

        best_clfs.append(get_best(clf, labels, features, features_list))

    best = 0
    best_clf = None
    for c in best_clfs:
        score = c[1]
        if score > best:
            best = score
            best_clf = c[0]

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

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    labels, features = targetFeatureSplit(data)

    tune_parameters(data_dict, features_list)