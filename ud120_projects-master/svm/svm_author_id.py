#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

print 'test'
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
from sklearn.svm import SVC

t0 = time()
clf = SVC(kernel  = 'rbf', C = 10000.0)  #'rbf'
clf.fit(features_train[:len(features_train)/100], labels_train[:len(labels_train)/100])
print "training time:" , round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "prediction time:" , round(time()-t0, 3), "s"

print clf.score(features_test, labels_test)

print pred[10], pred[26], pred[50]



#########################################################
### your code goes here ###

#########################################################


