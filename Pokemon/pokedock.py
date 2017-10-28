# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 14:04:20 2016

@author: Rupak Chakraborty
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn import cross_validation 

filepath = 'C:\\Users\\rupachak\\Desktop\\Kaggle Data\\Pokemon\\Pokemon.csv'
poke_frame = pd.read_csv(filepath)
label_encoder = LabelEncoder()
one_hot = OneHotEncoder()

poke_frame.drop(poke_frame.columns[[0,1]],axis=1,inplace=True)
poke_frame['Legendary'] = label_encoder.fit_transform(poke_frame['Legendary'].values)
poke_frame['Type 2'] = (label_encoder.fit_transform(poke_frame['Type 2'].values))
train_labels = poke_frame['Type 1'].values
del poke_frame['Type 1']
train_data = poke_frame.values

X_train,X_test,y_train,y_test = cross_validation.train_test_split(train_data,train_labels,test_size=0.2)

rf = RandomForestClassifier(n_estimators=51,max_depth=5)
ada = AdaBoostClassifier()
grad = GradientBoostingClassifier(n_estimators=51,max_depth=5)
bag = BaggingClassifier()

classifiers = [rf,ada,grad,bag]
classifier_names = ["Random Forests","Adaboost","Gradient Boost","Bagging"]

for classifier,classifier_name in zip(classifiers,classifier_names):
    classifier.fit(X_train,y_train)
    y_predict = classifier.predict(X_test)
    print 'For Classifier ',classifier_name,'\n'
    print metrics.classification_report(y_test,y_predict)
    print 'Accuracy ',metrics.accuracy_score(y_test,y_predict)