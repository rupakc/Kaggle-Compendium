# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 22:25:49 2016
Preprocessing Utilities for cleaning and processing of data
@author: Rupak Chakraborty
"""
import pandas as pd
from nltk.corpus import stopwords
import string 
from nltk.stem import PorterStemmer

stopword_list = set(stopwords.words("english")) 
punctuation_list = list(string.punctuation)
ps = PorterStemmer()

months_list = ["january","february","march","april","may","june","july","august",
		"september","october","november","december"]
  
digit_list = ["0","1","2","3","4","5","6","7","8","9"]
month_list_short = ["jan","feb","mar","apr","may","jun","jul","aug","sept","oct","nov","dec"]
emoticon_list = [":)",":(","^_^","-_-","<3",":D",":P",":/"] 

html_tag_list = ["&nbsp","&lt","&gt","&amp",";","<strong>","<em>","[1]","</strong>","</em>","<div>","</div>","<b>","</b>","[2]","[3]","...","[img]","[/img]","<u>","</u>","<p>","</p>","\n","\\t","<span>",
				"</span>","[Moved]","<br/>","<a>","</a>","&quot","<br>","<br />","Â","<a rel=\"nofollow\" class=\"ot-hashtag\"","&#39","<a","â€™","&#39;"] 
    
extend_punct_list = [' ',',',':',';','\'','\t','\n','?','-','$',"!!","?","w/","!","!!!","w/","'","RT","rt","@","#","/",":)",
":(",":D","^_^","^","...","&","\\",":","?","<",">","$","%","*","`","~","-","_",
"+","=","{","}","[","]","|","\"",",",";",")","(","r/","/u/","*","-"]

punctuation_list.extend(extend_punct_list)
#punctuation_list.remove(".")
months_list.extend(month_list_short)

"""
Given a string normalizes it, i.e. converts it to lowercase and strips it of extra spaces

Params:
--------
s - String which is to be normalized

Returns:
---------
String in the normalized form
"""
def normalize_string(s): 
    s = s.lower()
    s = s.strip()
    return s

"""
Given a list of strings normalizes the strings

Params:
-------
string_list - List containing the strings which are to be normalized

Returns:
---------
Returns a list containing the normalized string list
"""
def normalize_string_list(string_list):
    normalized_list = []
    for sentence in string_list:
        normalized_list.append(normalize_string(sentence))
    return normalized_list

"""
Given a string and a separator splits up the string in the tokens

Params:
--------
s - string which has to be tokenized
separator - separator based on which the string is to be tokenized

Returns:
---------
A list of words in the sentence based on the separator
"""

def tokenize_string(s,separator):
    word_list = list([])
    if isinstance(s,basestring):
        word_list = s.split(separator)
    return word_list

"""
Given a list of sentences tokenizes each sentence in the list

Params:
--------
string_list - List of sentences which have to be tokenized
separator - Separator based on which the sentences have to be tokenized
"""

def tokenize_string_list(string_list,separator):
    tokenized_sentence_list = []
    for sentence in string_list:
        sentence = sentence.encode("ascii","ignore")
        tokenized_sentence_list.append(tokenize_string(sentence,separator))
    return tokenized_sentence_list
        
"""
Given a string containing stopwords removes all the stopwords

Params:
--------

s - String containing the stopwords which are to be removed

Returns:
---------

String sans the stopwords
"""

def remove_stopwords(s): 
    
    s = s.lower()
    removed_string = ''
    words = s.split()
    for word in words:
        if word not in stopword_list:
            removed_string = removed_string + word.strip() + " "
            
    return removed_string.strip()

"""
Given a list of sentences and a filename, writes the sentences to the file

Params:
--------
sentence_list - List of sentences which have to be written to the file
filename - File to which the sentences have to be written

Returns:
---------
Nothing quite just writes the sentences to the file
"""
def write_sentences_to_file(sentence_list,filename): 
    
   write_file = open(filename,'w') 
    
   for sentence in sentence_list:
       write_file.write(encode_ascii(sentence) + '\n')
       write_file.flush() 
        
   write_file.close()

"""
Removes all the punctuations from a given string

Params:
--------

s - String containing the possible punctuations

Returns:
--------

String without the punctuations (including new lines and tabs)
"""

def remove_punctuations(s): 
    
    s = s.lower()
    s = s.strip()
    for punctuation in punctuation_list:
        s = s.replace(punctuation,' ')
    
    return s.strip()

"""
Strips a given string of HTML tags

Params:
--------

s - String from which the HTML tags have to be removed

Returns:
---------

String sans the HTML tags
"""
    
def remove_html_tags(s):
    
    for tag in html_tag_list:
        s = s.replace(tag,' ')
    return s

"""
Given a string removes all the digits from them 

Params:
-------

s - String from which the digits need to be removed

Returns:
---------

String without occurence of the digits
"""
    
def remove_digits(s):
    
    for digit in digit_list:
        s = s.replace(digit,'')
        
    return s

"""
Given a string returns all occurences of a month from it

Params:
--------

s - String containing possible month names

Returns:
--------

String wihtout the occurence of the months
"""
        
def remove_months(s): 
    
    s = s.lower()
    words = s.split()
    without_month_list = [word for word in words if word not in months_list]
    month_clean_string = "" 
    
    for word in without_month_list:
        month_clean_string = month_clean_string + word + " "
    
    return month_clean_string.strip()

"""
Checks if a given string contains all ASCII characters

Params:
-------

s - String which is to be checked for ASCII characters

Returns:
--------

True if the string contains all ASCII characters, False otherwise
"""
    
def is_ascii(s):
    if isinstance(s,basestring):
        return all(ord(c) < 128 for c in s)
    return False

"""
Given a string encodes it in ascii format

Params:
--------
s - String which is to be encoded

Returns:
--------
String encoded in ascii format
"""

def encode_ascii(s):
    return s.encode('ascii','ignore')

"""
Stems each word of a given sentence to it's root word using Porters Stemmer

Params:
--------

sentence - String containing the sentence which is to be stemmed

Returns:
---------

Sentence where each word has been stemmed to it's root word
"""
    
def stem_sentence(sentence):
    
    words = sentence.split()
    stemmed_sentence = ""
    for word in words:
        try:
            if is_ascii(word):
                stemmed_sentence = stemmed_sentence + ps.stem_word(word) + " "
        except:
            pass
        
    return stemmed_sentence.strip()

"""
Given a string removes urls from the string

Params:
--------
s - String containing urls which have to be removed

Returns:
--------

String without the occurence of the urls
"""

def remove_url(s):
    s = s.lower()
    words = s.split()
    without_url = ""
    for word in words:
        if word.count('http:') == 0 and word.count('https:') == 0 and word.count('ftp:') == 0 and word.count('www.') == 0 and word.count('.com') == 0 and word.count('.ly') == 0 and word.count('.st') == 0:
            without_url = without_url + word + " " 
            
    return without_url.strip()

"""
Given a string removes all the words whose length is less than 3

Params:
--------
s - String from which small words have to be removed.

Returns:
---------

Returns a string without occurence of small words
"""

def remove_small_words(s):
    words = s.split()
    clean_string = ""
    for word in words:
        if len(word) >= 3:
            clean_string = clean_string + word + " "
            
    return clean_string.strip()

"""
Defines the pipeline for cleaning and preprocessing of text

Params:
--------
s - String containing the text which has to be preprocessed

Returns:
---------
String which has been passed through the preprocessing pipeline
"""
    
def text_clean_pipeline(s):
    
    s = remove_url(s)
    s = remove_punctuations(s)
    s = remove_html_tags(s)
    s = remove_stopwords(s)
    s = remove_months(s)
    s = remove_digits(s)
    #s = stem_sentence(s)
    s = remove_small_words(s)
    
    return s

"""
Given a list of sentences processes the list through the pre-preprocessing pipeline and returns the list

Params:
--------
sentence_list - List of sentences which are to be cleaned

Returns:
---------

The cleaned and pre-processed sentence list
"""

def text_clean_pipeline_list(sentence_list):
    
    clean_sentence_list = list([])
    
    for s in sentence_list:

        s = remove_digits(s)
        s = remove_punctuations(s)
        s = remove_stopwords(s)
        s = remove_months(s)
        s = remove_small_words(s)
        #s = encode_ascii(s)
        s = remove_url(s)
        s = stem_sentence(s)

        clean_sentence_list.append(s)
    return clean_sentence_list

"""
Given a excel filepath and a corresponding sheetname reads it and converts it into a dataframe

Params:
--------
filename - Filepath containing the location and name of the file
sheetname - Name of the sheet containing the data

Returns:
---------
pandas dataframe containing the data from the excel file
"""

def get_dataframe_from_excel(filename,sheetname):
    
    xl_file = pd.ExcelFile(filename)
    data_frame = xl_file.parse(sheetname)
    
    return data_frame
