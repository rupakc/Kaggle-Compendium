# -*- coding: utf-8 -*-
"""
Created on Thu Jan 07 22:14:45 2016
Baseline for Kaggle Competition of Forest Cover Prediction
@author: Rupak Chakraborty
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn import metrics
from sklearn import cross_validation 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import KernelCenterer 
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding

filename = "Forest Cover Prediction/train.csv"
forestFrame = pd.read_csv(filename)
forestFrame = forestFrame.iloc[np.random.permutation(len(forestFrame))]
target_labels = forestFrame["Cover_Type"]

del forestFrame["Cover_Type"]
del forestFrame["Id"]

standard_scaler = StandardScaler()
normal_scaler = Normalizer()
min_max_scaler = MinMaxScaler()
max_abs_scaler = MaxAbsScaler()
kernel_center = KernelCenterer() 

preprocessors = [standard_scaler,normal_scaler,min_max_scaler,max_abs_scaler,kernel_center]
preprocessors_type = ["Standard Scaler","Normal Scaler","MinMaxScaler","MaxAbsScaler","Kernel Centerer"]


for preprocess, name in zip(preprocessors,preprocessors_type): 
    
    print "-------------------------------------\n"
    print "For Preprocessor : ",preprocess
    print "--------------------------------------\n" 
    
    data = preprocess.fit_transform(forestFrame.values)
    train_data,test_data,train_labels,test_labels = cross_validation.train_test_split(data,target_labels.values,test_size=0.3)
    
    rf = RandomForestClassifier(n_estimators=101)
    ada = AdaBoostClassifier(n_estimators=101)
    bagging = BaggingClassifier(n_estimators=101)
    gradBoost = GradientBoostingClassifier(n_estimators=101)
    
    classifiers = [rf,ada,bagging,gradBoost]
    classifier_names = ["Random Forests","Adaboost","Bagging","Gradient Boost"]
    
    for classifier,classifier_name in zip(classifiers,classifier_names):
        
        classifier.fit(train_data,train_labels)
        predicted_labels = classifier.predict(test_data)
        print "----------------------------------\n"
        print "Accuracy for ",classifier_name, " : ",metrics.accuracy_score(test_labels,predicted_labels)
        print "Confusion Matrix for ",classifier_name, " :\n ",metrics.confusion_matrix(test_labels,predicted_labels)
        print "----------------------------------\n"
