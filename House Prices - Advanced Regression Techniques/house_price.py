# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 16:39:14 2016
Baseline Ensemble Algorithms for the house price dataset
@author: Rupak Chakraborty
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer 
from sklearn import metrics
from sklearn import cross_validation

filepath = 'C:\\Users\\rupachak\\Desktop\\Kaggle Data\\House Prices - Advanced Regression Techniques\\train.csv'
house_frame = pd.read_csv(filepath)
del house_frame['Id']
label_encoder = LabelEncoder()
imputer = Imputer()

data_type_to_retain = house_frame['SaleType'].dtype
data_type_map = house_frame.columns.to_series().groupby(house_frame.dtypes).groups
list_of_objects = list(data_type_map[data_type_to_retain]) 

for column in list_of_objects:
    house_frame[column] = label_encoder.fit_transform(house_frame[column].values) 

output_values = house_frame['SalePrice'].values
del house_frame['SalePrice'] 
column_names = house_frame.columns
house_frame = imputer.fit_transform(house_frame.values)
house_frame = StandardScaler().fit_transform(house_frame)
output_values = StandardScaler().fit_transform(output_values)

X_train,X_test,y_train,y_test = cross_validation.train_test_split(house_frame,output_values.ravel(),test_size=0.2)

rf = RandomForestRegressor(n_estimators=51,max_depth=7,min_samples_leaf=5)
grad = GradientBoostingRegressor(n_estimators=51,max_depth=7,min_samples_leaf=5)
ada = AdaBoostRegressor()
bag = BaggingRegressor()

regressors = [rf,grad,ada,bag]
regressors_names = ['Random Forests','Gradient Boosting','Adaboost','Bagging']

for regressor,regressor_name in zip(regressors,regressors_names): 
    
    regressor.fit(X_train,y_train)
    y_predict = regressor.predict(X_test) 

    print "----------------------------------------------\n"
    print 'For the regressor - ',regressor_name
    print 'Mean Squared Error - ', metrics.mean_squared_error(y_test,y_predict)
    print 'Median Squared Error - ', metrics.median_absolute_error(y_test,y_predict)
    print 'Mean Absolute Error - ', metrics.mean_absolute_error(y_test,y_predict)
    print 'R2 Score - ', metrics.r2_score(y_test,y_predict)
    if regressor_name == "Random Forests":
        feature_importance_array = regressor.feature_importances_
        for index,value in enumerate(feature_importance_array):
            print column_names[index]," => ",value
    print "----------------------------------------------\n"