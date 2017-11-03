# -*- coding: utf-8 -*-
"""
Created on Sat Jan 02 17:28:40 2016

@author: Rupak Chakraborty
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn import cross_validation
from sklearn import metrics
import datetime

filename = "Bike Sharing/train.csv"
rental_data = pd.read_csv(filename)
date_format = "%Y-%m-%d %H:%M:%S"
del rental_data["casual"]
del rental_data["registered"]
rental_data["datetime"] = map(lambda x: datetime.datetime.strptime(x,date_format),rental_data["datetime"])

month_list = list([])
day_list = list([])
hour_list = list([])

for date in rental_data["datetime"]:
    
    month_list.append(date.month)
    day_list.append(date.day)
    hour_list.append(date.hour)

rental_data["month"] = np.array(month_list)
rental_data["day"] = np.array(day_list)
rental_data["hour"] = np.array(hour_list)

del rental_data["datetime"]
rental_data = rental_data.iloc[np.random.permutation(len(rental_data))]
rental_counts = rental_data["count"].values

train_data,test_data,train_counts,test_counts = cross_validation.train_test_split(rental_data.values,rental_counts,test_size=0.2)

rf = RandomForestRegressor(n_estimators=101)
ada = AdaBoostRegressor(n_estimators=101)
grad = GradientBoostingRegressor(n_estimators=101)
bagging = BaggingRegressor(n_estimators=101)
svr = SVR()

regressors = [rf,ada,grad,bagging,svr]
regressor_names = ["Random Forests","Adaboost Regressor","Gradient Boost Regressor","Bagging Regressor","Support Vector Regressor"]

for regressor,regressor_name in zip(regressors,regressor_names):
    
    regressor.fit(train_data,train_counts)
    predicted_counts = regressor.predict(test_data)
    
    print "-----------------------------------------\n"
    print "Mean Absolute Error for ",regressor_name," : ",metrics.mean_absolute_error(test_counts,predicted_counts)
    print "Median Absolute Error for ",regressor_name," : ",metrics.median_absolute_error(test_counts,predicted_counts)
    print "Mean Squared Error for ",regressor_name," : ",metrics.mean_squared_error(test_counts,predicted_counts)
    print "R2 Score for ",regressor_name, " : ",metrics.r2_score(test_counts,predicted_counts)
    print "Explained Variance for ",regressor_name," : ",metrics.explained_variance_score(test_counts,predicted_counts)
    print "----------------------------------------\n"