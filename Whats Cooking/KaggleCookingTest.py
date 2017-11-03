# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 15:49:23 2015
Test Class for the Kaggle Cooking Competition
@author: Rupak Chakraborty
"""
import json
import ClassificationUtils

def getJSONData(filename):
    
    f = open(filename,"r")
    contents = f.read()
    jsonData = json.loads(contents)
    return jsonData

def getIngredientList(jsonData):
    ingredient_list = list([])
    for recipe in jsonData:
        s = ""
        if "ingredients" in recipe:
            for ingredient in recipe["ingredients"]:
                s = s + " " + ingredient
            ingredient_list.append(s)
            
    return ingredient_list

def testProcessingPipeline(filename):
    
    jsonData = getJSONData(filename)
    ingredient_list = getIngredientList(jsonData) 
    
    tfidf = ClassificationUtils.load_classifier("tfidf.pickle")
    bag_of_words = ClassificationUtils.load_classifier("bag_of_words.pickle")
    adaboost = ClassificationUtils.load_classifier("ada_idf_cook.pickle")
    randomfor = ClassificationUtils.load_classifier("rf_idf_cook.pickle")
    bagging = ClassificationUtils.load_classifier("bagging_idf_cook.pickle")
    
    test_data_tfidf = tfidf.transform(ingredient_list)
    test_data_bag = bag_of_words.transform(ingredient_list)
    
    adaboost.predict(test_data_bag)
    adaboost.predict(test_data_tfidf)
    
    randomfor.predict(test_data_bag)
    randomfor.predict(test_data_tfidf)
    
    bagging.predict(test_data_bag)
    bagging.predict(test_data_tfidf)

if __name__ == "__main__":
    
    testfilename = "test.json"
    testProcessingPipeline(testfilename)
    