# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 23:21:29 2016
Defines a set of utility functions to be used for prediction
@author: Rupak Chakraborty
"""
import math
from trip import Trip
from gift import Gift
import random 
import time
import pandas as pd
import operator 

RADIUS_EARTH = 6773
NORTH_POLE_LAT = 90
NORTH_POLE_LONG = 0
EMPTY_SLEIGH_WEIGHT = 10
SLEIGH_CAPACITY = 1000
random.seed(time.time())
gift_filename = "Santa's Stolen Sleigh/gifts.csv"

"""
Calculates the haversine distance between two given points 
The two points are the values of the latitude and longitude
in degrees

Params:
--------
lat_first - Latitude of the first point
long_first - Longitude of the first point
lat_second - Latitude of second point
long_second - Longitude of the second point

Returns:
--------- 
The haversine distance between the two given points i.e. a float 

"""
def haversineDistance(lat_first,long_first,lat_second,long_second):
    
    lat_first = math.radians(lat_first)
    long_first = math.radians(long_first)
    lat_second = math.radians(lat_second)
    long_second = math.radians(long_second)
    
    sine_squared_lat = math.pow(math.sin((lat_first-lat_second)/2.0),2.0)
    sine_squared_long = math.pow(math.sin((long_first-long_second)/2.0),2.0)
    cos_lat_term = math.cos(lat_first)*math.cos(lat_second)*sine_squared_long
    total_term = cos_lat_term + sine_squared_lat
    
    distance = 2*RADIUS_EARTH*math.asin(math.sqrt(total_term))
    
    return distance 
    
"""
Defines the fitness function for the trip list i.e. all deliveries
The total fitness is defined as the weighted sum of distances

Params:
--------
trip_list: A List of trips which Santa needs to take (A list containing the trip object)

Returns:
---------
Total Cost of the given trip list (i.e. Fitness) 
"""
def tripFitness(trip_list): 
    
    total_cost = 0     
    
    for trip in trip_list:
        total_cost = total_cost + trip.trip_cost 
        
    return total_cost 

"""
Given a list of gifts calculates the cost of the trip (i.e. Weighted Distance)

Params:
-------- 
gift_list: A list of gifts in the order in which they have to be delivered

Returns:
--------- 

Cost of the trip with the given order of gifts (i.e. A Floating point number)
"""
def tripCost(gift_list):
    
    gift_size = len(gift_list)
    initial_gift_weight = tripWeightUtil(gift_list,0,gift_size-1)
    weighted_distance = initial_gift_weight*haversineDistance(NORTH_POLE_LAT,NORTH_POLE_LONG,gift_list[0].latitude,gift_list[0].longitude)
    
    for i in range(gift_size-1):
        remaining_weight = tripWeightUtil(gift_list,i+1,gift_size-1)
        distance = haversineDistance(gift_list[i].latitude,gift_list[i].longitude,gift_list[i+1].latitude,gift_list[i+1].longitude)
        weighted_distance = weighted_distance + remaining_weight*distance
    
    returning_distance = haversineDistance(gift_list[gift_size-1].latitude,gift_list[gift_size-1].longitude,NORTH_POLE_LAT,NORTH_POLE_LONG)
    weighted_distance = weighted_distance + EMPTY_SLEIGH_WEIGHT*returning_distance
    
    return weighted_distance  
    
"""
Utility function to calculate the cumulative weight of gifts in a given range
Both ends of the range are included

Params:
-------- 

gift_list : List of gift objects
start_index : Starting index for gift list
end_index : Ending index of the gift list

Returns:
---------

Returns the sum of weights in a given range

"""        
def tripWeightUtil(gift_list,start_index,end_index):
    
    total_weight = 0 
    
    while start_index <= end_index:
        total_weight = total_weight + gift_list[start_index].weight
        start_index = start_index + 1
    
    return total_weight 
    
"""
Applies the mutation operator on trip list i.e. swaps two trips

Params:
-------
trip_list: List containing the trips taken by Santa

Returns:
--------
A new list containing the trip list with values swapped
""" 

def mutateTripList(trip_list):
    
    i,j = generateSwapIndices(len(trip_list))
    temp = trip_list[i]
    trip_list[i] = trip_list[j]
    trip_list[j] = temp
    
    return trip_list 

"""
Applies the mutation operator on the gift list i.e. swaps two gifts in a list

Params:
-------
gift_list: List containing the gifts taken by Santa

Returns:
--------
A new list containing the gift list with values swapped
"""
 
def mutateGiftList(gift_list): 
    
    i,j = generateSwapIndices(len(gift_list))
    temp = gift_list[i]
    gift_list[i] = gift_list[j]
    gift_list[j] = temp
    
    return gift_list

"""
Utility function to generate two distinct random integers from zero to a given range

Params:
--------
max_size: Integer containing the maximum limit for generation of the random integers

Returns:
--------
Two distinct random integers between 0 and a given max_size
"""
def generateSwapIndices(max_size):
    
    a = random.randint(0,max_size-1)
    b = random.randint(0,max_size-1) 
    
    while b != a:
        b = random.randint(0,max_size) 
        
    return a,b
"""
Returns the dataFrame containing the gift information

Params:
------- 
String containing the filename from which the information is to be extracted

Returns:
--------

Pandas Dataframe object containing the gift information
"""
def getGiftList(filename):
    
    giftFrame = pd.read_csv(filename)
    gift_list = list([])
    
    for i in range(len(giftFrame)): 
        
        gift_series = giftFrame.iloc[i]
        gift = Gift(gift_series.GiftId,gift_series.Latitude,gift_series.Longitude,gift_series.Weight) 
        
        gift_list.append(gift)
        
    return gift_list;

"""
Sorts a given map by the values and returns a list containing th sorted tuples
"""
def sortMapByValues(map_to_sort): 
    
    sorted_map = sorted(map_to_sort.items(), key=operator.itemgetter(1),reverse=False)
    return sorted_map
    
"""
Sorts the given population by its fitness value

Params:
-------
initial_population: List containing the initial population

Returns:
--------
List of tuples containing the indices of the initial population and its fitness
"""
def sortPopulationByFitness(initial_population):
    
    i = 0;
    fitness_population_map = {} 
    
    for trip_gene in initial_population: 
        fitness_population_map[i] = tripFitness(trip_gene)
        i = i + 1

    ordered_fitness_list = sortMapByValues(fitness_population_map)
    
    return ordered_fitness_list
"""
Given all the trips in a list returns the one with the maximum cost and its index

Params:
---------
trip_list: List of trips to be taken for delivery

Returns:
--------
The trip with the maximum cost and its corresponding index
"""
def maximumTripCost(trip_list):
    index = 0
    max_trip = trip_list[0] 
    
    for i,trip in enumerate(trip_list): 
        
        if trip.trip_cost > max_trip:
            max_trip = trip.trip_cost
            index = i
            
    return index,trip

    
    
        