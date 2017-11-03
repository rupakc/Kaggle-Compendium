# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 09:37:18 2015
Text Filelds of JSON - 
request_text
request_title
request_text_edit_aware
requester_subreddits_at_request (JSONArray)
Count of subreddit field - 
requester_number_of_subreddits_at_request
@author: Rupak Chakraborty
"""

import ClassificationUtils
import scipy

filename = "Random Acts of Pizza/train.json"
jsonData = ClassificationUtils.getJSONData(filename)
subreddits_pizza = set([])
subreddits_non_pizza = set([]) 
subreddit_pizza_map = {}
subreddit_non_pizza_map = {}
pizza_title_list = []
non_pizza_title_list = []
pizza_text_list = []
non_pizza_text_list = []

pizza_request_suffix_map =  {"requester_upvotes_minus_downvotes_at_request":[],
        "requester_number_of_subreddits_at_request":[],
        "requester_upvotes_plus_downvotes_at_request":[],
        "requester_account_age_in_days_at_request":[],
        "requester_days_since_first_post_on_raop_at_request":[],
        "requester_number_of_comments_at_request":[],
        "requester_number_of_comments_in_raop_at_request":[],
        "requester_number_of_posts_at_request":[],
        "requester_number_of_posts_on_raop_at_request":[] 
        }

non_pizza_request_suffix_map =  {"requester_upvotes_minus_downvotes_at_request":[],
        "requester_number_of_subreddits_at_request":[],
        "requester_upvotes_plus_downvotes_at_request":[],
        "requester_account_age_in_days_at_request":[],
        "requester_days_since_first_post_on_raop_at_request":[],
        "requester_number_of_comments_at_request":[],
        "requester_number_of_comments_in_raop_at_request":[],
        "requester_number_of_posts_at_request":[],
        "requester_number_of_posts_on_raop_at_request":[] 
        }

pizza_retrival_suffix_map = {"requester_upvotes_minus_downvotes_at_retrieval":[],
        "requester_upvotes_plus_downvotes_at_retrieval":[],
        "number_of_downvotes_of_request_at_retrieval":[],
        "number_of_upvotes_of_request_at_retrieval":[],
        "request_number_of_comments_at_retrieval":[],
        "requester_account_age_in_days_at_retrieval":[],
        "requester_days_since_first_post_on_raop_at_retrieval":[],
        "requester_number_of_comments_at_retrieval":[],
        "requester_number_of_comments_in_raop_at_retrieval":[],
        "requester_number_of_posts_at_retrieval":[],
        "requester_number_of_posts_on_raop_at_retrieval" : []
        }

non_pizza_retrival_suffix_map = {"requester_upvotes_minus_downvotes_at_retrieval":[],
        "requester_upvotes_plus_downvotes_at_retrieval":[],
        "number_of_downvotes_of_request_at_retrieval":[],
        "number_of_upvotes_of_request_at_retrieval":[],
        "request_number_of_comments_at_retrieval":[],
        "requester_account_age_in_days_at_retrieval":[],
        "requester_days_since_first_post_on_raop_at_retrieval":[],
        "requester_number_of_comments_at_retrieval":[],
        "requester_number_of_comments_in_raop_at_retrieval":[],
        "requester_number_of_posts_at_retrieval":[],
        "requester_number_of_posts_on_raop_at_retrieval" : []
        }
        
def prettyPrintSortedDict(dictionary):
    
    for key in dictionary:
        print key[0] , " : ", key[1]
        
def processRequestNumericalData(post): 
    
    if post["requester_received_pizza"]:
        
        pizza_request_suffix_map["requester_upvotes_minus_downvotes_at_request"].append(post["requester_upvotes_minus_downvotes_at_request"])
        pizza_request_suffix_map["requester_number_of_subreddits_at_request"].append(post["requester_number_of_subreddits_at_request"])
        pizza_request_suffix_map["requester_upvotes_plus_downvotes_at_request"].append(post["requester_upvotes_plus_downvotes_at_request"])
        pizza_request_suffix_map["requester_account_age_in_days_at_request"].append(post["requester_account_age_in_days_at_request"])
        pizza_request_suffix_map["requester_days_since_first_post_on_raop_at_request"].append(post["requester_days_since_first_post_on_raop_at_request"])
        pizza_request_suffix_map["requester_number_of_comments_at_request"].append(post["requester_number_of_comments_at_request"])
        pizza_request_suffix_map["requester_number_of_comments_in_raop_at_request"].append(post["requester_number_of_comments_in_raop_at_request"])
        pizza_request_suffix_map["requester_number_of_posts_at_request"].append(post["requester_number_of_posts_at_request"])
        pizza_request_suffix_map["requester_number_of_posts_on_raop_at_request"].append(post["requester_number_of_posts_on_raop_at_request"]) 
        
    else:
        
        non_pizza_request_suffix_map["requester_upvotes_minus_downvotes_at_request"].append(post["requester_upvotes_minus_downvotes_at_request"])
        non_pizza_request_suffix_map["requester_number_of_subreddits_at_request"].append(post["requester_number_of_subreddits_at_request"])
        non_pizza_request_suffix_map["requester_upvotes_plus_downvotes_at_request"].append(post["requester_upvotes_plus_downvotes_at_request"])
        non_pizza_request_suffix_map["requester_account_age_in_days_at_request"].append(post["requester_account_age_in_days_at_request"])
        non_pizza_request_suffix_map["requester_days_since_first_post_on_raop_at_request"].append(post["requester_days_since_first_post_on_raop_at_request"])
        non_pizza_request_suffix_map["requester_number_of_comments_at_request"].append(post["requester_number_of_comments_at_request"])
        non_pizza_request_suffix_map["requester_number_of_comments_in_raop_at_request"].append(post["requester_number_of_comments_in_raop_at_request"])
        non_pizza_request_suffix_map["requester_number_of_posts_at_request"].append(post["requester_number_of_posts_at_request"])
        non_pizza_request_suffix_map["requester_number_of_posts_on_raop_at_request"].append(post["requester_number_of_posts_on_raop_at_request"]) 

def processRetrievalNumericalData(post):
    
    if post["requester_received_pizza"]:
        
        pizza_retrival_suffix_map["requester_upvotes_minus_downvotes_at_retrieval"].append(post["requester_upvotes_minus_downvotes_at_retrieval"])
        pizza_retrival_suffix_map["requester_upvotes_plus_downvotes_at_retrieval"].append(post["requester_upvotes_plus_downvotes_at_retrieval"])
        pizza_retrival_suffix_map["number_of_downvotes_of_request_at_retrieval"].append(post["number_of_downvotes_of_request_at_retrieval"])
        pizza_retrival_suffix_map["number_of_upvotes_of_request_at_retrieval"].append(post["number_of_upvotes_of_request_at_retrieval"])
        pizza_retrival_suffix_map["request_number_of_comments_at_retrieval"].append(post["request_number_of_comments_at_retrieval"])
        pizza_retrival_suffix_map["requester_account_age_in_days_at_retrieval"].append(post["requester_account_age_in_days_at_retrieval"])
        pizza_retrival_suffix_map["requester_days_since_first_post_on_raop_at_retrieval"].append(post["requester_days_since_first_post_on_raop_at_retrieval"]) 
        pizza_retrival_suffix_map["requester_number_of_comments_at_retrieval"].append(post["requester_number_of_comments_at_retrieval"])
        pizza_retrival_suffix_map["requester_number_of_comments_in_raop_at_retrieval"].append(post["requester_number_of_comments_in_raop_at_retrieval"])
        pizza_retrival_suffix_map["requester_number_of_posts_at_retrieval"].append(post["requester_number_of_posts_at_retrieval"])
        pizza_retrival_suffix_map["requester_number_of_posts_on_raop_at_retrieval"].append(post["requester_number_of_posts_on_raop_at_retrieval"])
    
    else:
        non_pizza_retrival_suffix_map["requester_upvotes_minus_downvotes_at_retrieval"].append(post["requester_upvotes_minus_downvotes_at_retrieval"])
        non_pizza_retrival_suffix_map["requester_upvotes_plus_downvotes_at_retrieval"].append(post["requester_upvotes_plus_downvotes_at_retrieval"])
        non_pizza_retrival_suffix_map["number_of_downvotes_of_request_at_retrieval"].append(post["number_of_downvotes_of_request_at_retrieval"])
        non_pizza_retrival_suffix_map["number_of_upvotes_of_request_at_retrieval"].append(post["number_of_upvotes_of_request_at_retrieval"])
        non_pizza_retrival_suffix_map["request_number_of_comments_at_retrieval"].append(post["request_number_of_comments_at_retrieval"])
        non_pizza_retrival_suffix_map["requester_account_age_in_days_at_retrieval"].append(post["requester_account_age_in_days_at_retrieval"])
        non_pizza_retrival_suffix_map["requester_days_since_first_post_on_raop_at_retrieval"].append(post["requester_days_since_first_post_on_raop_at_retrieval"]) 
        non_pizza_retrival_suffix_map["requester_number_of_comments_at_retrieval"].append(post["requester_number_of_comments_at_retrieval"])
        non_pizza_retrival_suffix_map["requester_number_of_comments_in_raop_at_retrieval"].append(post["requester_number_of_comments_in_raop_at_retrieval"])
        non_pizza_retrival_suffix_map["requester_number_of_posts_at_retrieval"].append(post["requester_number_of_posts_at_retrieval"])
        non_pizza_retrival_suffix_map["requester_number_of_posts_on_raop_at_retrieval"].append(post["requester_number_of_posts_on_raop_at_retrieval"])
    

for post in jsonData:
    
    if "requester_received_pizza" in post:  
        
        processRequestNumericalData(post)
        processRetrievalNumericalData(post) 
        
        pizza_status = post["requester_received_pizza"] 
        
        if pizza_status:
            pizza_title_list.append(post["request_title"])
            pizza_text_list.append(post["request_text_edit_aware"])
        else:
            non_pizza_title_list.append(post["request_title"])
            non_pizza_text_list.append(post["request_text_edit_aware"])
            
        for subreddit in post["requester_subreddits_at_request"]: 
                
            if pizza_status and post["requester_number_of_subreddits_at_request"] > 0: 
                    
                    if subreddit.lower() not in subreddits_pizza: 
                        
                        subreddit_pizza_map[subreddit.lower()] = 1
                        subreddits_pizza.add(subreddit.lower())
                    else:
                        subreddit_pizza_map[subreddit.lower()] = subreddit_pizza_map[subreddit.lower()] + 1
            else: 
                
                if post["requester_number_of_subreddits_at_request"] > 0: 
                    
                    if subreddit.lower() not in subreddits_non_pizza: 
                        
                        subreddit_non_pizza_map[subreddit.lower()] = 1
                        subreddits_non_pizza.add(subreddit.lower())
                    else:
                        subreddit_non_pizza_map[subreddit.lower()] = subreddit_non_pizza_map[subreddit.lower()] + 1
                        
subreddit_pizza_map = sorted(subreddit_pizza_map.items(),key=lambda x:x[1],reverse=True)
subreddit_non_pizza_map = sorted(subreddit_non_pizza_map.items(),key=lambda x:x[1],reverse=True)

for key in pizza_request_suffix_map:
    print key , " : ", scipy.mean(pizza_request_suffix_map[key])
    print key , " : ", scipy.median(pizza_request_suffix_map[key])
    print key , " : ", max(pizza_request_suffix_map[key])
    print key , " : ", min(pizza_request_suffix_map[key])

for key in non_pizza_request_suffix_map:
    print key , " : ", scipy.mean(non_pizza_request_suffix_map[key])
    print key , " : ", scipy.median(non_pizza_request_suffix_map[key])
    print key , " : ", max(non_pizza_request_suffix_map[key])
    print key , " : ", min(non_pizza_request_suffix_map[key])

for key in pizza_retrival_suffix_map:
    print key , " : ", scipy.mean(pizza_retrival_suffix_map[key])
    print key , " : ", scipy.median(pizza_retrival_suffix_map[key])
    print key , " : ", max(pizza_retrival_suffix_map[key])
    print key , " : ", min(pizza_retrival_suffix_map[key])

for key in non_pizza_retrival_suffix_map:
    print key , " : ", scipy.mean(non_pizza_retrival_suffix_map[key])
    print key , " : ", scipy.median(non_pizza_retrival_suffix_map[key])
    print key , " : ", max(non_pizza_retrival_suffix_map[key])
    print key , " : ", min(non_pizza_retrival_suffix_map[key])



                