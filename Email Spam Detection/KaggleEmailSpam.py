# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 16:17:36 2015
Kaggle Email Spam Detection Using a set of 100 features
@author: Rupak Chakraborty
"""
import numpy as np
import ClassificationUtils
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

svm = SVC()
bnb = BernoulliNB(alpha=0.2)
mnb = MultinomialNB(alpha=0.4)
gnb = GaussianNB()
rf = RandomForestClassifier(n_jobs=4,n_estimators=17)
knn = KNeighborsClassifier(n_neighbors=5)

trainDataFile = "train.csv"
trainLabelFile = "train_labels.csv"

fLabel = open(trainLabelFile,"r")
labels = fLabel.read()
class_labels = labels.split()
train_set = np.zeros((len(class_labels),100))
ClassificationUtils.populateNumpyData(trainDataFile,train_set)

svm.fit(train_set,class_labels)
predict_svm = svm.predict(train_set)
print "Accuracy For SVM - ", metrics.accuracy_score(class_labels,predict_svm)
print "----------------- Classification Report for SVM ---------------------"
print metrics.classification_report(class_labels,predict_svm)
ClassificationUtils.save_classifier("svm_email.pickle",svm)

bnb.fit(train_set,class_labels)
predict_bnb = bnb.predict(train_set)
print "Accuracy For Bernoulli - ", metrics.accuracy_score(class_labels,predict_bnb)

print "----------------- Classification Report for Bernoulli ---------------------"
print metrics.classification_report(class_labels,predict_bnb)
ClassificationUtils.save_classifier("bnb_email.pickle",bnb)

gnb.fit(train_set,class_labels)
predict_gnb = gnb.predict(train_set)
print "Accuracy For Gaussian - ", metrics.accuracy_score(class_labels,predict_gnb)
print "----------------- Classification Report for Gaussian ---------------------"
print metrics.classification_report(class_labels,predict_gnb)
ClassificationUtils.save_classifier("gnb_email.pickle",gnb)

mnb.fit(train_set,class_labels)
predict_mnb = mnb.predict(train_set)
print "Accuracy For Multinomial - ", metrics.accuracy_score(class_labels,predict_mnb)
print "----------------- Classification Report for Multinomial ---------------------"
print metrics.classification_report(class_labels,predict_mnb)
ClassificationUtils.save_classifier("mnb_email.pickle",mnb)

rf.fit(train_set,class_labels)
predict_rf = rf.predict(train_set)
print "Accuracy For Random Forest - ", metrics.accuracy_score(class_labels,predict_rf)
print "----------------- Classification Report for Random Forest ---------------------"
print metrics.classification_report(class_labels,predict_rf)
ClassificationUtils.save_classifier("rf_email.pickle",rf)

knn.fit(train_set,class_labels)
predict_knn = knn.predict(train_set)
print "Accuracy for KNN (K = 5) - ", metrics.accuracy_score(class_labels,predict_knn)
print "----------------- Classification Report for KNN (K = 5) ---------------------"
print metrics.classification_report(class_labels,predict_knn)
ClassificationUtils.save_classifier("knn_email.pickle",knn)
