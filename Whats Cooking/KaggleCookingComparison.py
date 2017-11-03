# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 13:20:45 2015
Code for Kaggle What's Cooking Competition
It uses the following classifiers with tf-idf,hashvectors and bag_of_words approach
1. Adaboost
2. Extratrees
3. Bagging
4. Random Forests
@author: Rupak Chakraborty
"""

import numpy as np
import time
import json
import ClassificationUtils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics

# Create the feature extractors 

bag_of_words = CountVectorizer(stop_words='english')
tfidf = TfidfVectorizer(stop_words='english')
hashvec = HashingVectorizer(stop_words='english')

# Create the Classifier objects

adaboost = AdaBoostClassifier()
randomforest = RandomForestClassifier()
extratrees = ExtraTreesClassifier()
bagging = BaggingClassifier()

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
ingredient_list = list([])
c = 0 

print "Size of the data set : ", len(jsonData)
print "Starting Loading of Data Set...."
start = time.time()

for recipe in jsonData: 
    
    if "cuisine" in recipe: 
        s = ""
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
            
        s = s + " " + ingredient 
        
    ingredient_list.append(s)
    
end = time.time()
print "Time Taken to Load the Dataset : ",end-start

for cuisine in cuisine_set:
    cuisine_numerical_map[cuisine] = c
    c = c+1
c = 0
for ingredient in ingredient_set:
    ingredient_numerical_map[ingredient] = c
    c = c+1

print "Starting Feature Extracting ......"
start = time.time()
train_labels = np.zeros(len(ingredient_list))
train_data_tfidf = tfidf.fit_transform(ingredient_list)
train_data_hash = hashvec.fit_transform(ingredient_list)
train_data_bag = bag_of_words.fit_transform(ingredient_list)

c = 0

for recipe in jsonData:
    if "cuisine" in recipe:
        train_labels[c] = cuisine_numerical_map[recipe["cuisine"]]
    c = c+1
end = time.time()

print "Time Taken to Train Extract Different Features : ", end-start

test_labels = train_labels[1:30000]
test_data_tfidf = tfidf.transform(ingredient_list[1:30000])
test_data_hash = hashvec.transform(ingredient_list[1:30000])
test_data_bag = bag_of_words.transform(ingredient_list[1:30000])

print "Starting Training of Models for Hash Vectorizer Feature....."
start = time.time()
adaboost.fit(train_data_bag,train_labels)
randomforest.fit(train_data_bag,train_labels)
extratrees.fit(train_data_bag,train_labels)
bagging.fit(train_data_bag,train_labels)
end=time.time()
print "Time Taken to train all Ensemble Models : ", end-start

print "Starting Prediction of Test Labels ...."
start = time.time()
ada_predict = adaboost.predict(test_data_bag)
rf_predict = randomforest.predict(test_data_bag)
extree_predict = extratrees.predict(test_data_bag)
bagging_predict = bagging.predict(test_data_bag)
end = time.time()
print "Time Taken to Test the models : ", end-start 

print "Accuracy of AdaBoost Algorithm : ", metrics.accuracy_score(test_labels,ada_predict)
print "Accuracy of Random Forests : ", metrics.accuracy_score(test_labels,rf_predict)
print "Accuracy of Extra Trees : ", metrics.accuracy_score(test_labels,extree_predict)
print "Accuracy of Bagging : ", metrics.accuracy_score(test_labels,bagging_predict)

# Saving the tf-idf model and classifiers

ClassificationUtils.save_classifier("ada_bag_cook.pickle",adaboost)
ClassificationUtils.save_classifier("rf_bag_cook.pickle",randomforest)
ClassificationUtils.save_classifier("extree_bag_cook.pickle",extratrees)
ClassificationUtils.save_classifier("bagging_bag_cook.pickle",bagging)
ClassificationUtils.save_classifier("bag_of_words.pickle",tfidf)

def printIngredientDistribution():           
    print "----------- Distribution of the Recipe Ingredients ------------------"
    for key in ingredient_map.keys():
        print  key, " : " ,ingredient_map[key]

def printCuisineDistribution():
    print "----------- Distribution of the Cuisines ------------------"
    for key in cuisine_map.keys():
        print  key, " : " ,cuisine_map[key] 
 

    
