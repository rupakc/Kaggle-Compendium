# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 23:03:00 2015
Defines the data model for Random Acts of Pizza
@author: Rupak Chakraborty
"""
import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 
from sklearn import cross_validation
from sklearn import metrics
import ClassificationUtils 
import time
import nltk
from nltk.tokenize import word_tokenize
 

filename = "Random Acts of Pizza/train.json"
class_map = {True:1,False:0}
jsonData = pd.read_json(filename)
jsonData = jsonData.iloc[np.random.permutation(len(jsonData))]

requester_pizza_status = np.array(map(lambda x: class_map[x],jsonData["requester_received_pizza"]))
class_labels = requester_pizza_status
data = np.zeros((len(jsonData),18))

#Function to Extract POS tag counts from a given text

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


# Extract Text features  

request_text__data = list(jsonData["request_text_edit_aware"])
request_text_title_data = list (jsonData["request_title"])
clean_text_data = list([])
clean_title_data = list([])

print "Starting feature loading and cleaning ..."
start = time.time()

for i in range(len(request_text__data)): 
    
    title_string = ClassificationUtils.textCleaningPipeline(request_text_title_data[i])
    text_string = ClassificationUtils.textCleaningPipeline(request_text__data[i])
    clean_text_data.append(text_string)
    clean_title_data.append(title_string)     
    
end = time.time()
print "Time taken to load and clean text features : ", end-start   

# Extract whole features 

number_of_downvotes_of_request_at_retrieval = np.array(jsonData["number_of_downvotes_of_request_at_retrieval"],dtype=float)
number_of_upvotes_of_request_at_retrieval = np.array(jsonData["number_of_upvotes_of_request_at_retrieval"],dtype=float)
request_number_of_comments_at_retrieval = np.array(jsonData["request_number_of_comments_at_retrieval"],dtype=float)
requester_number_of_subreddits_at_request = np.array(jsonData["requester_number_of_subreddits_at_request"],dtype=float)

whole_features = [number_of_downvotes_of_request_at_retrieval,number_of_upvotes_of_request_at_retrieval,\
                  request_number_of_comments_at_retrieval,requester_number_of_subreddits_at_request] 
                  
# Extract pairwise different features

requester_account_age_in_days_at_request = np.array(jsonData["requester_account_age_in_days_at_request"],dtype=float)
requester_account_age_in_days_at_retrieval = np.array(jsonData["requester_account_age_in_days_at_retrieval"],dtype=float) 

requester_days_since_first_post_on_raop_at_request = np.array(jsonData["requester_days_since_first_post_on_raop_at_request"],dtype=float)
requester_days_since_first_post_on_raop_at_retrieval = np.array(jsonData["requester_days_since_first_post_on_raop_at_retrieval"],dtype=float)

requester_number_of_comments_at_request = np.array(jsonData["requester_number_of_comments_at_request"],dtype=float)
requester_number_of_comments_at_retrieval = np.array(jsonData["requester_number_of_comments_at_retrieval"],dtype=float)

requester_number_of_comments_in_raop_at_request = np.array(jsonData["requester_number_of_comments_in_raop_at_request"],dtype=float)
requester_number_of_comments_in_raop_at_retrieval = np.array(jsonData["requester_number_of_comments_in_raop_at_retrieval"],dtype=float)

requester_number_of_posts_at_request = np.array(jsonData["requester_number_of_posts_at_request"],dtype=float)
requester_number_of_posts_at_retrieval = np.array(jsonData["requester_number_of_posts_at_retrieval"],dtype=float)

requester_number_of_posts_on_raop_at_request = np.array(jsonData["requester_number_of_posts_on_raop_at_request"],dtype=float)
requester_number_of_posts_on_raop_at_retrieval = np.array(jsonData["requester_number_of_posts_on_raop_at_retrieval"],dtype=float)

requester_upvotes_minus_downvotes_at_request = np.array(jsonData["requester_upvotes_minus_downvotes_at_request"],dtype=float)
requester_upvotes_minus_downvotes_at_retrieval = np.array(jsonData["requester_upvotes_minus_downvotes_at_retrieval"],dtype=float)

requester_upvotes_plus_downvotes_at_request = np.array(jsonData["requester_upvotes_plus_downvotes_at_request"],dtype=float)
requester_upvotes_plus_downvotes_at_retrieval = np.array(jsonData["requester_upvotes_plus_downvotes_at_retrieval"],dtype=float)

request_features = [requester_account_age_in_days_at_request,requester_days_since_first_post_on_raop_at_request\
,requester_number_of_comments_at_request,requester_number_of_comments_in_raop_at_request,requester_number_of_posts_at_request\
,requester_number_of_posts_on_raop_at_request,requester_upvotes_minus_downvotes_at_request,requester_upvotes_plus_downvotes_at_request]

retrieval_features = [requester_account_age_in_days_at_retrieval,requester_days_since_first_post_on_raop_at_retrieval\
,requester_number_of_comments_at_retrieval,requester_number_of_comments_in_raop_at_retrieval,requester_number_of_posts_at_retrieval\
,requester_number_of_posts_on_raop_at_retrieval,requester_upvotes_minus_downvotes_at_retrieval,requester_upvotes_plus_downvotes_at_retrieval]


#Extracting and organizing the data in a numpy array  

print "Starting feature organization and POS tagging"
start = time.time() 

for i in range(len(data)): 
    
    feature_row = [] 
    
    for whole in whole_features: 
        
        feature_row.append(whole[i]) 

    for index,retrieval in enumerate(retrieval_features): 
        
        difference = retrieval[i] - request_features[index][i]
        difference = ((difference + 1.0)/(request_features[index][i] + 1.0))*100.0 
        
        if math.isinf(difference) or math.isnan(difference): 
            difference = 1.0
            
        feature_row.append(difference) 
        
    text_pos_tags = getNounAdjVerbs(clean_text_data[i])
    title_post_tags = getNounAdjVerbs(clean_title_data[i])
    total_pos_tag_count = text_pos_tags + title_post_tags 
    
    for tag_count in total_pos_tag_count:
        feature_row.append(tag_count) 
        
    data[i,:] = feature_row
    
end = time.time()
print "Time Taken to extract all features : ", end-start

train_data,test_data,train_label,test_label = cross_validation.train_test_split(data,class_labels,test_size=0.3)

# Initializing the classifiers 

rf = RandomForestClassifier(n_estimators=101)
ada = AdaBoostClassifier(n_estimators=101)
gradboost = GradientBoostingClassifier(n_estimators=101)
svm = SVC()
gnb = GaussianNB()

classifiers = [rf,ada,gradboost,svm,gnb]
classifier_names = ["Random Forests","AdaBoost","Gradient Boost","SVM","Gaussian NB"]

print "Starting Classification Performance Cycle ..."
start = time.time()

for classifier,classifier_name in zip(classifiers,classifier_names):
    
    classifier.fit(train_data,train_label)
    predicted_label = classifier.predict(test_data) 
    
    print "--------------------------------------------------------\n"
    print "Accuracy for ",classifier_name, " : ",metrics.accuracy_score(test_label,predicted_label)
    print "Confusion Matrix for ",classifier_name, " :\n ",metrics.confusion_matrix(test_label,predicted_label)
    print "Classification Report for ",classifier_name, " : \n",metrics.classification_report(test_label,predicted_label)
    print "--------------------------------------------------------\n" 
    
end = time.time()
print "Time Taken for classification and performance Metrics calculation : ",end-start