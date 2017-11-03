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

filename = "Random Acts of Pizza/train.json"
class_map = {True:1,False:0}
jsonData = pd.read_json(filename)
jsonData = jsonData.iloc[np.random.permutation(len(jsonData))]

requester_pizza_status = np.array(map(lambda x: class_map[x],jsonData["requester_received_pizza"]))
class_labels = requester_pizza_status
data = np.zeros((len(jsonData),12))

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
        
    data[i,:] = feature_row

train_data,test_data,train_label,test_label = cross_validation.train_test_split(data,class_labels,test_size=0.3)

rf = RandomForestClassifier(n_estimators=101)
ada = AdaBoostClassifier(n_estimators=101)
gradboost = GradientBoostingClassifier(n_estimators=101)
svm = SVC()
gnb = GaussianNB()

classifiers = [rf,ada,gradboost,svm,gnb]
classifier_names = ["Random Forests","AdaBoost","Gradient Boost","SVM","Gaussian NB"]

for classifier,classifier_name in zip(classifiers,classifier_names):
    
    classifier.fit(train_data,train_label)
    predicted_label = classifier.predict(test_data)
    print "Accuracy for ",classifier_name, " : ",metrics.accuracy_score(test_label,predicted_label)

            
        
        



