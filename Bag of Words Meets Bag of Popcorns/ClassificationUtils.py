# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 18:08:00 2015
A Set of utility functions for saving and loading classifiers
@author: Rupak Chakraborty
"""
import pickle
import math
import json
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stopword_list = set(stopwords.words("english"))
ps = PorterStemmer() 

def populateNumpyData(filename,train_set): 
    
    train_file = open(filename,"r")
    data = train_file.read()
    train_data = data.split()
    c = 0
    for feature_row in train_data:
        features = feature_row.split(",")
        for i in range(len(features)):
            k = float(features[i])
            if math.isnan(k) or math.isinf(k):
                k = 0.01
            train_set[c][i] = k
        c = c + 1

def convertJSONArrayToSpaceDelimitedString(jsonArray): 
    
    sentence = ""
    for item in jsonArray: 
        sentence = sentence + " " + item
        
    return item.strip() 
    
def getJSONData(filename):
    
    f = open(filename,"r")
    contents = f.read()
    jsonData = json.loads(contents)
    return jsonData
    
def save_classifier(filename,classifier):
    
    save_file = open(filename,"wb")
    pickle.dump(classifier,save_file)
    save_file.close
    
def load_classifier(filename): 
    
    classifier_file = open(filename,"rb")
    classifier = pickle.load(classifier_file)
    classifier_file.close()
    return classifier 
    
def is_ascii(s):
    return all(ord(c) < 128 for c in s)
    
def stemSentence(sentence):
    words = word_tokenize(sentence)
    stemmed_sentence = ""
    for word in words:
        try:
            if is_ascii(word):
                stemmed_sentence = stemmed_sentence + ps.stem_word(word) + " "
        except:
            pass
        
    return stemmed_sentence.strip() 
    
def cleanHTML(sentence): 
    
    tag_list = ["<br />","1","2","3","4","5","6","7","8","9","0"] 
    
    for tag in tag_list:
        sentence = sentence.replace(tag,"")
    
    return sentence
    
def remove_punctuations(sentence): 
    
    punctuations = list(string.punctuation)
    punctuations.append('\n')
    punctuations.append('\r')
    punctuations.append('\r\n') 
    
    for punct in punctuations:
        sentence = sentence.replace(punct,"") 
        
    return sentence.strip()

def removeStopWords(sentence): 
    
    stopwordfree_sentence = ""    
    words = word_tokenize(sentence)
    without_stopwords = [word for word in words if word not in stopword_list]
    
    for word in without_stopwords:
        stopwordfree_sentence = stopwordfree_sentence + " " + word 
        
    return stopwordfree_sentence.strip()

def textCleaningPipeline(sentence):
    
    sentence = sentence.lower()
    sentence = cleanHTML(sentence)
    sentence = remove_punctuations(sentence)
    sentence = removeStopWords(sentence)
    sentence = stemSentence(sentence)
    
    return sentence