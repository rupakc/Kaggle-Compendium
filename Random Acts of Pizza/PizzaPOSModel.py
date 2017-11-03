# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 20:22:10 2015

@author: Rupak Chakraborty
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
import ClassificationUtils 
import time
import nltk
from nltk.tokenize import word_tokenize

filename = "Random Acts of Pizza/train.json"
class_map = {True:1,False:0}
jsonData = pd.read_json(filename)
request_text__data = list(jsonData["request_text_edit_aware"])
request_text_title_data = list (jsonData["request_title"])
request_number_of_subreddits = list(jsonData["requester_number_of_subreddits_at_request"])
requester_pizza_status = list(jsonData["requester_received_pizza"])
title_list = list([])
text_list = list([])
TRAIN_SIZE = 3500

print "Starting Loading and Cleaning of Dataset ... "
start = time.time() 

for i in range(len(request_number_of_subreddits)): 
    
    title_string = ClassificationUtils.textCleaningPipeline(request_text_title_data[i])
    text_string = ClassificationUtils.textCleaningPipeline(request_text__data[i])
    title_list.append(title_string)
    text_list.append(text_string)
    requester_pizza_status[i] = class_map[requester_pizza_status[i]] 
    
end = time.time()
print "Time Taken to Load and Clean Dataset : ", end-start

data = np.zeros((len(requester_pizza_status),6))

def generatePOSTagFeatures(title_string_list,text_string_list):
    for i in range(len(title_string_list)):
        title = title_string_list[i]
        text = text_string_list[i]
        title_tuple = getNounAdjVerbs(title)
        text_tuple = getNounAdjVerbs(text)
        data[i,:] = np.array(title_tuple + text_tuple)

def getNounAdjVerbs(text): 

    words = word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    nouns = 0
    verbs = 0
    adj = 0
    for token in pos_tags:
        k = token[1]
        if k == "NN" or k == "NNP" or k == "NNS" or k == "NNPS":
            nouns = nouns + 1
        elif k == "JJ" or k == "JJR" or k == "JJS":
            adj = adj + 1
        elif k == "VB" or k == "VBD" or k == "VBG" or k == "VBN" or k == "VBP" or k == "VBZ":
            verbs = verbs + 1
    return nouns,adj,verbs

#Populating the data set with features 

start=time.time() 
print "Starting Extraction of POS tag Features...."
generatePOSTagFeatures(title_list,text_list)
end = time.time()
print "Time Taken to Extract Post Tag Feature : ",end-start 

#Splitting the data into train and test set 

train_data = data[:TRAIN_SIZE]
train_labels = requester_pizza_status[:TRAIN_SIZE]
test_data = data[TRAIN_SIZE:]
test_labels = requester_pizza_status[TRAIN_SIZE:] 

#Initializing Classifiers

svm = SVC(kernel='rbf')
mnb = MultinomialNB()
rf = RandomForestClassifier(n_estimators=51)
ada = AdaBoostClassifier(n_estimators=100)

classifiers = [svm,mnb,rf,ada]
classifier_names = ["SVM","Multinomial NB","Random Forest","AdaBoost"]

for classifier,classifier_name in zip(classifiers,classifier_names):
    classifier.fit(train_data,train_labels)
    predicted_labels = classifier.predict(test_data)
    print "Accuracy for ",classifier_name," : ",metrics.accuracy_score(test_labels,predicted_labels)