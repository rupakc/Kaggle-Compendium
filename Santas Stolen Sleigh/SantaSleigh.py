# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 13:58:24 2016
Main module to carry out the optimization of routes for Santa's Gift Delivery
Basic Benchmark Created but the mutation operator still needs work
@author: Rupak Chakraborty
"""

import math
from trip import Trip
from gift import Gift
import random
import time
import pandas as pd
import SantaUtil

SLEIGH_CAPACITY = 1000
INTIAL_POPULATION_SIZE = 100
MAX_ITERATIONS = 10000
TOP_K = 10
random.seed(time.time())
gift_filename = "Santa's Stolen Sleigh/gifts.csv"
GIFT_MUTATION_PROBABILITY = 0.7
TRIP_CROSSOVER_PROBABILITY = 0.67 #These can be rather should be fine-tuned for hyperparameter optimization

start = time.time()
print "Initiating Dataset Loading ...."
gift_list = SantaUtil.getGiftList(gift_filename)
end = time.time()
print "Time Taken to load dataset : ",end-start

"""
Generates the initial population of trip routes to be taken

Params:
--------
gift_list: List containing the list of gifts to be delivered
initial_population_size: Integer containing the number of initial populations to be generated

Returns:
A List of trips denoting the order in which the gifts are to be delivered
"""
def generateInitialTripListPopulation(gift_list,initial_population_size):

    count = 0
    trip_list = list([])
    master_trip_population = list([])
    gift_trip_list = list([])

    while count < initial_population_size:

        random.shuffle(gift_list)
        total_weight = 0
        i = 0
        trip_list = list([])

        while i < len(gift_list):

            while i < len(gift_list) and (total_weight + gift_list[i].weight) <= SLEIGH_CAPACITY:

                gift_trip_list.append(gift_list[i])
                total_weight = total_weight + gift_list[i].weight
                i = i + 1

            trip_order = Trip(gift_trip_list,SantaUtil.tripCost(gift_trip_list))
            total_weight = 0
            gift_trip_list = list([])
            trip_list.append(trip_order)

        count = count+1
        master_trip_population.append(trip_list)

    return master_trip_population

print "Starting Generating Initial Population...."
start = time.time()
initial_population = generateInitialTripListPopulation(gift_list,INTIAL_POPULATION_SIZE)
end = time.time()
print "Total Time Taken for Creating Initial Pool : ",end-start

ordered_fitness_list = SantaUtil.sortPopulationByFitness(initial_population)
best_solution_index = ordered_fitness_list[0][0]
best_solution_cost = ordered_fitness_list[0][1]
"""
Generates the best solution from the initial population by using elitist policy and
mutation operator on the order of gifts

Params:
--------
initial_population: List containing the list of initial seed genes
ordered_fitness_list: List containing the sorted tuples of the initial_population sorted by fitness
max_iterations: The maximum number of iterations to run before convergence

Returns:
--------

The index of the best solution and the total weighted cost of the trip
"""
def generateBestSolution(initial_population,ordered_fitness_list,max_iterations):

    iteration = 0
    while iteration < max_iterations:

       iteration = iteration + 1
       gift_mutation = random.random()
      
       best_solution_index = ordered_fitness_list[0][0]
       best_solution_cost = ordered_fitness_list[0][1]
       count = 0 
       
       while count <= 10:
           
           count = count + 1
           
           if gift_mutation >= GIFT_MUTATION_PROBABILITY:
    
               gene_index = random.randint(len(initial_population)-TOP_K-1,len(initial_population)-1)
               chosen_trip_index = ordered_fitness_list[gene_index][0]
               chosen_trip_list = initial_population[chosen_trip_index]
               index,chosen_trip = SantaUtil.maximumTripCost(chosen_trip_list)
               swapped_gift_list = SantaUtil.mutateGiftList(chosen_trip.gift_list)
               temp_trip_cost = SantaUtil.tripCost(swapped_gift_list)
    
               if temp_trip_cost < chosen_trip.trip_cost:
    
                   chosen_trip.gift_list = swapped_gift_list
                   initial_population[chosen_trip_index] = chosen_trip
                   ordered_fitness_list = SantaUtil.sortPopulationByFitness(initial_population)
    
           if gift_mutation < GIFT_MUTATION_PROBABILITY:
    
               gene_index = random.randint(0,TOP_K+1)
               chosen_trip_index = ordered_fitness_list[gene_index][0]
               chosen_trip_list = initial_population[chosen_trip_index]
               index,chosen_trip = SantaUtil.maximumTripCost(chosen_trip_list)
               swapped_gift_list = SantaUtil.mutateGiftList(chosen_trip.gift_list)
               temp_trip_cost = SantaUtil.tripCost(swapped_gift_list)
    
               if temp_trip_cost < chosen_trip.trip_cost:
    
                   chosen_trip.gift_list = swapped_gift_list
                   initial_population[chosen_trip_index] = chosen_trip
                   ordered_fitness_list = SantaUtil.sortPopulationByFitness(initial_population) 
                   
    return best_solution_index,best_solution_cost

print generateBestSolution(initial_population,ordered_fitness_list,MAX_ITERATIONS)
