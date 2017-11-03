# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 22:46:08 2015
request_text
request_title
request_text_edit_aware
requester_subreddits_at_request (JSONArray)
requester_number_of_subreddits_at_request
requester_received_pizza
@author: Rupak Chakraborty
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
import ClassificationUtils 
import time

filename = "Random Acts of Pizza/train.json"
class_map = {True:1,False:0}
jsonData = pd.read_json(filename)
request_text__data = list(jsonData["request_text_edit_aware"])
request_text_title_data = list (jsonData["request_title"])
request_number_of_subreddits = list(jsonData["requester_number_of_subreddits_at_request"])
requester_pizza_status = list(jsonData["requester_received_pizza"])
requester_subreddits_at_request = list(jsonData["requester_subreddits_at_request"])
data_set = [] 
TRAIN_SIZE = 3500

print "Starting Loading of Dataset... "
start = time.time() 

for i in range(len(requester_subreddits_at_request)): 
    
    jsonArray = requester_subreddits_at_request[i]
    subreddit_string = ClassificationUtils.convertJSONArrayToSpaceDelimitedString(jsonArray)
    title_string = ClassificationUtils.textCleaningPipeline(request_text_title_data[i])
    text_string = ClassificationUtils.textCleaningPipeline(request_text__data[i])
    final_string = title_string + " " + text_string + " " + subreddit_string
    data_set.append(final_string)
    requester_pizza_status[i] = class_map[requester_pizza_status[i]] 
    
end = time.time()
print "Time Taken to Load Dataset : ", end-start

#Splitting the data into train and test set

train_data = data_set[:TRAIN_SIZE]
test_data = data_set[TRAIN_SIZE:]
train_labels = requester_pizza_status[:TRAIN_SIZE]
test_labels = requester_pizza_status[TRAIN_SIZE:] 

#Initializing the feature vectors 

count_vec = CountVectorizer()
tf_idf = TfidfVectorizer()

#Extracting features for both train and test set 

print "Starting Feature Extraction..."
start = time.time()

train_data_bag = count_vec.fit_transform(train_data)
test_data_bag = count_vec.transform(test_data)

train_data_tfidf = tf_idf.fit_transform(train_data)
test_data_tfidf = tf_idf.transform(test_data)

end = time.time()
print "Time Taken for Feature Extraction : ", end-start

train_feature_list = [train_data_bag,train_data_tfidf]
test_feature_list = [test_data_bag,test_data_tfidf]
feature_names = ["Bag Of Words", "Tf-Idf"]

#Initializing the classifiers 

rf = RandomForestClassifier(n_estimators=51,random_state=1)
svm = SVC(kernel="linear",probability=True)
mnb = MultinomialNB(fit_prior=True)
ada = AdaBoostClassifier(random_state=1)

#Creating an estimator list for Voting Classifiers 

classifier_names = ["Random Forests","SVM","Multinomial NB","Adaboost"]
classifiers = [rf,svm,mnb,ada]
estimator_list = zip(classifier_names,classifiers)

print "Starting Training of Classifiers and predicting on both set of features..." 
start = time.time()

for train_set,test_set,feature in zip(train_feature_list,test_feature_list,feature_names):
    print "For Feature Extraction : ",feature
    for classifier,classifier_name in zip(classifiers,classifier_names):
        classifier.fit(train_set,train_labels)
        predict = classifier.predict(test_set)
        print "-----------------------------------------------------------------"
        print "Accuracy for ", classifier_name , " : ", metrics.accuracy_score(test_labels,predict)
        print "Confusion Matrix for ", classifier_name, " :\n ", metrics.confusion_matrix(test_labels,predict)
        print "Classification Report for ", classifier_name, ":\n",metrics.classification_report(test_labels,predict)
        print "-----------------------------------------------------------------" 
        
end = time.time()
print "Time Taken for Entire Train and Test Cycle : ", end-start         

# Defining the voting classifiers 

def voting():
    voting_clf = VotingClassifier(estimators=estimator_list,voting='hard')
    for train_set,test_set,feature in zip(train_feature_list,test_feature_list,feature_names):
        print "For Feature Extraction : ",feature
        voting_clf.fit(train_set,train_labels)
        predict = voting_clf.predict(test_set)
        print "Accuracy for Voting Classifiers : ", metrics.accuracy_score(test_labels,predict)
        print "Confusion Matrix for Voting Classifier : ", metrics.confusion_matrix(test_labels,predict)
        print "Classification Report for Voting Classifiers : ",metrics.classification_report(test_labels,predict)

print "Starting Voting of Classifiers.... "
start = time.time()
voting()
end = time.time()
print "Time Taken for Voting Classifiers : ", end-start
