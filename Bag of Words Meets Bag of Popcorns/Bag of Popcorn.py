# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 21:28:54 2015 

Solution to the Kaggle Competition Bag of Words Meet Bag of Popcorns
Feature Extraction Involves - 
1. CountVectorizer
2. TfIdf Vectorizer 

Classifiers Included - 
SVM (Linear)
Multinomial NB
Perceptron
BernoulliNB
KNN (k=5)
Random Forests

Performance Metrics used during testing - 
Accuracy Score
Confusion Matrix
Classification Report (which includes precision,recall and f1-score)
Matthews Correlation Coefficient
Area Under the Curve
@author: Rupak Chakraborty
"""
import pandas as pd
import ClassificationUtils
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import time

sentiment_data = []
sentiment_labels = []
TRAIN_SIZE = 20000
filename = "Bag of Words Meet Bag of Popcorn (Google Word2Vec)/labeledTrainData.tsv"
data = pd.read_csv(filename,sep="\t")

# Preprocessing the Data
print "Starting Preprocessing Data...."
start = time.time() 

for label,review in zip(data["sentiment"],data["review"]):
    sentiment_data.append(ClassificationUtils.textCleaningPipeline(review))
    sentiment_labels.append(label) 
    
end = time.time()
print "Taken Taken for Data Preprocessing : ",end-start

#Separating the Training and Test Labels 
    
train_labels = sentiment_labels[:TRAIN_SIZE]
test_labels = sentiment_labels[TRAIN_SIZE:]
train_data = sentiment_data[:TRAIN_SIZE]
test_data = sentiment_data[TRAIN_SIZE:]

#Initializing Feature Extractors 

count_vec = CountVectorizer()
tfidf = TfidfVectorizer()

#Extracting Training and Test Features 

print "Starting Feature Extraction.."
start = time.time()

train_set_bag = count_vec.fit_transform(train_data)
train_set_tfidf = tfidf.fit_transform(train_data)
test_set_bag = count_vec.transform(test_data)
test_set_tfidf = tfidf.transform(test_data)

end = time.time()
print "Time Taken For Feature Extraction : ", end-start

# Initializing Classifiers 

perceptron = Perceptron()
mnb = MultinomialNB()
bnb = BernoulliNB()
rf = RandomForestClassifier(n_estimators=91)
knn = KNeighborsClassifier(n_neighbors=3)

# Listing Features and Classifiers

test_feature_list = [test_set_bag,test_set_tfidf]
train_feature_list = [train_set_bag,train_set_tfidf]
feature_name_list = ["Bag of Words","Tf-Idf"]
classifier_name_list = ["Perceptron","Multinomial NB","Bernoulli NB","Random Forest","KNN(K=5)"]
classifier_list = [perceptron,mnb,bnb,knn,rf]

# Iterating for the feature set and the list of classifiers to generate the results 

start = time.time()
for train,test,feature_name in zip(train_feature_list,test_feature_list,feature_name_list):
    
    print "---- Results for Feature ------ : ",feature_name
    
    for classifier,classifier_name in zip(classifier_list,classifier_name_list):
        
        classifier.fit(train,train_labels)
        predicted_labels = classifier.predict(test)
        print "-------------------------------------------------\n"
        print "Accuracy for ", classifier_name, ": ", metrics.accuracy_score(test_labels,predicted_labels)
        print "Classification Report for ", classifier_name, ":\n", metrics.classification_report(test_labels,predicted_labels)
        print "Confusion Matrix for ", classifier_name, ":\n", metrics.confusion_matrix(test_labels,predicted_labels)
        print "Matthews Correlation Coefficient for ", classifier_name, ":\n ", metrics.matthews_corrcoef(test_labels,predicted_labels)
        print "Area Under Curve for ", classifier_name, ":\n",metrics.roc_auc_score(test_labels,predicted_labels)
        print "-------------------------------------------------\n"
        
end = time.time()
print "Time Taken for Entire Classification : ", end-start