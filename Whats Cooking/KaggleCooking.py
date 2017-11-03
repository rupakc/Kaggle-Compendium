# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 00:54:59 2015
Code for Kaggle What's Cooking Competition
It uses the following classifiers with one-hot vector encoding
1. Bernoulli Naive Bayes
2. Gaussian Naive Bayes
3. Multinomial Naive Bayes
4. Random Forests
@author: Rupak Chakraborty
"""

import json
import numpy as np
import time
import pylab as pl
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
import ClassificationUtils

filepath = "train.json"
f = open(filepath,"r")

content = f.read()
jsonData = json.loads(content)
cuisine_set = set([])
ingredient_set = set([])
cuisine_map = {}
cuisine_numerical_map = {}  
ingredient_numerical_map = {}
ingredient_map = {}
c = 0 

print "Size of the data set : ", len(jsonData)

for recipe in jsonData:
    if "cuisine" in recipe:
        if recipe["cuisine"] in cuisine_set:
            cuisine_map[recipe["cuisine"]] = cuisine_map[recipe["cuisine"]] + 1
        else:
            cuisine_map[recipe["cuisine"]] = 1
            cuisine_set.add(recipe["cuisine"])
    for ingredient in recipe["ingredients"]:
        if ingredient in ingredient_set:
            ingredient_map[ingredient] = ingredient_map[ingredient] + 1
        else:
            ingredient_map[ingredient] = 1
            ingredient_set.add(ingredient)

for cuisine in cuisine_set:
    cuisine_numerical_map[cuisine] = c
    c = c+1
c = 0
for ingredient in ingredient_set:
    ingredient_numerical_map[ingredient] = c
    c = c+1

train_data = np.zeros((len(jsonData),len(ingredient_set)))
train_label = np.zeros((len(jsonData)))
c = 0 

start = time.time()
for recipe in jsonData:
    if "cuisine" in recipe:
        train_label[c] = cuisine_numerical_map[recipe["cuisine"]]
    if "ingredients" in recipe:
        for ingredient in recipe["ingredients"]:
            train_data[c][ingredient_numerical_map[ingredient]] = 1
    c = c + 1
end = time.time() 

print "Time Taken to Extract Features : ", end-start 


test_data = train_data[0:3000]
test_label = train_label[0:3000] 

bnb = BernoulliNB()
gnb = GaussianNB()
mnb = MultinomialNB()
randfor = RandomForestClassifier(n_jobs=4,n_estimators=23)
supvec = SVC() 

start = time.time()
bnb.fit(train_data,train_label)
gnb.fit(train_data,train_label)
mnb.fit(train_data,train_label)
randfor.fit(train_data,train_label)

end = time.time()
ClassificationUtils.save_classifier("bnb_cook.pickle",bnb)
ClassificationUtils.save_classifier("gnb_cook.pickle",gnb)
ClassificationUtils.save_classifier("mnb_cook.pickle",mnb)
ClassificationUtils.save_classifier("rf_cook.pickle",randfor)

print randfor.feature_importances_
print "Time Taken to Train Models : ", end-start

start = time.time()
bernoullipredict = bnb.predict(test_data)
gaussianpredict = gnb.predict(test_data)
multinomialpredict = mnb.predict(test_data)
randforestpredict = randfor.predict(test_data)

print "--------- Classification Report for Bernoulli Bayes ---------------"
print metrics.classification_report(test_label,bernoullipredict)
print "--------- Classification Report for Gaussian Bayes ---------------"
print metrics.classification_report(test_label,gaussianpredict)
print "--------- Classification Report for Multinomial Bayes ---------------"
print metrics.classification_report(test_label,multinomialpredict)
print "--------- Classification Report for Random Forest ---------------"
print metrics.classification_report(test_label,randforestpredict)

print "Bernoulli - ", metrics.accuracy_score(test_label,bernoullipredict)
print "Gaussian - ", metrics.accuracy_score(test_label,gaussianpredict)
print "Multinomial - ", metrics.accuracy_score(test_label,multinomialpredict)
print "Random Forest - ", metrics.accuracy_score(test_label,randforestpredict)
end = time.time()

print "Time Taken to Test Models : ", end-start

def plotCuisineDistribution(): # TODO - Add parameters here
    pl.figure(1)
    pl.pie(cuisine_map.values(),labels=cuisine_map.keys())
    pl.show()

def prettyPrintDict(dictionary): 
    for key in dictionary:
        print key , ": " ,dictionary[key]
