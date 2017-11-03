# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 18:08:00 2015
A Set of utility functions for saving and loading classifiers
@author: Rupak Chakraborty
"""
import pickle
import math

def populateNumpyData(filename,train_set):
    train_file = open(filename,"r")
    data = train_file.read()
    train_data = data.split()
    c = 0
    for feature_row in train_data:
        features = feature_row.split(",")
        for i in range(len(features)):
            k = float(features[i])
            if math.isnan(k) or math.isinf(k):
                k = 0.01
            train_set[c][i] = k
        c = c + 1
        
def save_classifier(filename,classifier):
    
    save_file = open(filename,"wb")
    pickle.dump(classifier,save_file)
    save_file.close
    
def load_classifier(filename):
    classifier_file = open(filename,"rb")
    classifier = pickle.load(classifier_file)
    classifier_file.close()
    return classifier