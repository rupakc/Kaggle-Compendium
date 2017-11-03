# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 17:48:58 2015
Baseline Solution for the Kaggle Competition Machine Learning for Disaster
@author: Rupak Chakraborty
"""

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

train_file = "Titanic - Machine Learning For Disaster/train.csv" 
test_file = "Titanic - Machine Learning For Disaster/test.csv" #Replace with your own filepaths
gender_class_file = "Titanic - Machine Learning For Disaster/genderclassmodel.csv"

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
test_label_data = pd.read_csv(gender_class_file)
columns_to_delete = ['PassengerId','Name','Ticket','Cabin'] 
sex_map = {"male":0,"female":1}
embarked_map = {"Q":0,"S":1,"C":2}

train_class_labels = np.array(train_data["Survived"])
test_class_labels = np.array(test_label_data["Survived"])

del train_data["Survived"]

#Dropping unecessary info from the dataframe

for column in columns_to_delete:
    
    del train_data[column]
    del test_data[column]

#Filling in the missing values in the dataframe and converting categoricals to numbers

train_data["Age"].fillna(train_data.groupby("Sex")["Age"].transform("mean"),inplace=True)
train_data["Fare"].fillna(train_data.groupby("Sex")["Fare"].transform("mean"),inplace=True)

test_data["Age"].fillna(test_data.groupby("Sex")["Age"].transform("mean"),inplace=True)
test_data["Fare"].fillna(test_data.groupby("Sex")["Fare"].transform("mean"),inplace=True)

train_data["Embarked"].fillna('S',inplace=True)
test_data["Embarked"].fillna('S',inplace=True)

train_data["Sex"] = map(lambda x : sex_map[x],train_data["Sex"])
test_data["Sex"] = map(lambda x : sex_map[x],test_data["Sex"])

train_data["Embarked"] = map(lambda x : embarked_map[x],train_data["Embarked"])
test_data["Embarked"] = map(lambda x : embarked_map[x],test_data["Embarked"])

#Initializing Classifiers

rf = RandomForestClassifier(n_estimators=51)
ada = AdaBoostClassifier(n_estimators=51)
bag = BaggingClassifier(n_estimators=51)
gradboost = GradientBoostingClassifier()
classifier_list = [rf,ada,bag,gradboost]
classifier_names = ["Random Forests","AdaBoost","Bagging","Gradient Boost"] 

#Iterating over classifiers in order to find the performance metrics

for classifier,classifier_name in zip(classifier_list,classifier_names): 
    
    classifier.fit(train_data.values,train_class_labels)
    predicted_labels = classifier.predict(test_data.values)
    print "------------------------------------------------\n"
    print "Accuracy for ",classifier_name," : ",metrics.accuracy_score(test_class_labels,predicted_labels)
    print "Confusion Matrix for ",classifier_name," :\n",metrics.confusion_matrix(test_class_labels,predicted_labels)
    print "Classification Report for ",classifier_name, " :\n",metrics.classification_report(test_class_labels,predicted_labels)
    print "------------------------------------------------\n" 
    
voting_classifier = VotingClassifier(estimators=zip(classifier_names,classifier_list),voting='hard')
voting_classifier.fit(train_data.values,train_class_labels)
predicted_labels = voting_classifier.predict(test_data.values)
print "Accuracy for Voting Classifier : ",metrics.accuracy_score(test_class_labels,predicted_labels)
print "Confusion Matrix for Voting Classifier :\n ",metrics.confusion_matrix(test_class_labels,predicted_labels)
print "Classification Report for Voting Classifier : \n",metrics.classification_report(test_class_labels,predicted_labels)
