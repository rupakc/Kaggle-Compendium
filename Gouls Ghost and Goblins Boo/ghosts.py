# -*- coding: utf-8 -*-
"""
Created on Thurs Aug 17 09:23:27 2017

@author: Rupak Chakraborty
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

filepath = 'C:\\Users\\rupachak\\Desktop\\Kaggle Data\\Gouls Ghost and Goblins Boo\\train.csv' # Replace this with your own filepath
color_map = {'white':0,'black':1,'clear':2,'blue':3,'green':4,'blood':5}
train_df = pd.read_csv(filepath)
del train_df['id']
train_df['color'] = map(lambda x:color_map[x],train_df['color'].values)
train_labels = train_df['type'].values
train_labels = LabelEncoder().fit_transform(train_labels)
del train_df['type']

X_train,X_test,y_train,y_test = train_test_split(train_df.values,train_labels,test_size=0.2,random_state=42)
x_axis_variable = "bone_length"
y_axis_variable = "hair_length"
train_data = train_df[[x_axis_variable,y_axis_variable]].values
step_size = 0.02

rf = RandomForestClassifier(n_estimators=101,random_state=42,min_samples_split=3,min_samples_leaf=11,max_depth=5)
bag = BaggingClassifier(n_estimators=121,random_state=42)
grad = GradientBoostingClassifier(n_estimators=51,random_state=42,min_samples_leaf=3,min_samples_split=5,max_features=2)
SVM = svm.SVC()

classifiers = [rf,bag,grad,SVM]
classifier_names = ['Random Forests','Bagging','Gradient Boosting',' SVM ']


def generate_optimal_hyperparameter_grid_search(X_train, y_train, X_test, y_test):

    max_accuracy = 0
    optimal_configuration = dict({})

    for estimators in range(11,111,10):
        for samples_split in range(2,7,1):
            for min_samples_per_leaf in range(1,11,1):

                rf = RandomForestClassifier(n_estimators=estimators,random_state=42,min_samples_split=samples_split,min_samples_leaf=min_samples_per_leaf,criterion='entropy')
                rf.fit(X_train,y_train)
                predicted_values = rf.predict(X_test)
                temp_accuracy = metrics.accuracy_score(y_test,predicted_values)
                print temp_accuracy
                if temp_accuracy > max_accuracy:
                    max_accuracy= temp_accuracy
                    optimal_configuration["Estimators"] = estimators
                    optimal_configuration["Samples Per Split"] = samples_split
                    optimal_configuration["Min Samples Per Leaf"] = min_samples_per_leaf

    print "Optimal Configuration ", optimal_configuration
    print "Maximum Accuracy ", max_accuracy


def print_performance_metrics(classifiers, classifier_names, X_train, y_train, X_test, y_test):

    for classifier,classifier_name in zip(classifiers,classifier_names):
        classifier.fit(X_train,y_train.ravel())
        y_predict = classifier.predict(X_test)
        print 'For Classifier ',classifier_name,'\n'
        print metrics.classification_report(y_test,y_predict)
        print metrics.accuracy_score(y_test,y_predict)


def plot_decision_boundaries(classifier_list, classifier_name_list, train_data, train_labels, x_label, y_label, step_size=0.02):

    x_min, x_max = train_data[:, 0].min() - 1, train_data[:, 0].max() + 1
    y_min, y_max = train_data[:, 1].min() - 1, train_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    for index, classifier in enumerate(classifier_list):
        classifier.fit(train_data, train_labels)
        Z = classifier.predict(zip(xx.ravel(), yy.ravel()))
        Z = Z.reshape(xx.shape)
        plt.subplot(len(classifier_list),2, index + 1)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.contourf(xx, yy, Z, cmap=plt.cm.CMRmap, alpha=0.6)
        plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, cmap=plt.cm.gist_rainbow, alpha=0.3)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(str(classifier_name_list[index]))

    plt.show()

print_performance_metrics(classifiers,classifier_names,X_train,y_train,X_test,y_test)
plot_decision_boundaries(classifiers,classifier_names,train_data,train_labels,x_axis_variable,y_axis_variable)
#generate_optimal_hyperparameter_grid_search(X_train,y_train,X_test,y_test)
