# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 23:24:38 2016
Defines a trip object for the a specific trip
@author: Rupak Chakraborty
"""
class Trip():
    
    gift_list = list([])
    
    def __init__(self,gift_list,trip_cost):
        
        self.gift_list = gift_list
        self.trip_cost = trip_cost;
