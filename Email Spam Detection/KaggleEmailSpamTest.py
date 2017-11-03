# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 18:18:43 2015
Test Data for the Kaggle Email Spam Test Competition
@author: Rupak Chakraborty
"""

import numpy as np
import ClassificationUtils

test_filename = "test.csv"
num_features = 100
test_file = open(test_filename,"r")
test_data = test_file.read()
test_data = test_data.split()
test_set = np.zeros((len(test_data),num_features))
ClassificationUtils.populateNumpyData(test_filename,test_set)

svm = ClassificationUtils.load_classifier("svm_email.pickle")
rf = ClassificationUtils.load_classifier("rf_email.pickle")
bnb = ClassificationUtils.load_classifier("bnb_email.pickle") 
gnb = ClassificationUtils.load_classifier("gnb_email.pickle")
mnb = ClassificationUtils.load_classifier("mnb_email.pickle")

svm_predict = svm.predict(test_set)
rf_predict = rf.predict(test_set)
bnb_predict = bnb.predict(test_set)
gnb_predict = gnb.predict(test_set)
mnb_predict = mnb.predict(test_set)