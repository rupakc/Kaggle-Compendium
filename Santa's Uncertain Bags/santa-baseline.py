import numpy as np
import pandas as pd
import commonutils
import constants
import copy

np.random.seed(42)


def mutate_solutions(solution_list,mutation_rate = constants.mutation_probability):
    for random_solution in solution_list:
        bag_list = random_solution.bag_list
        for random_bag in bag_list:
            gift_weight_list = random_bag.gift_weight_list
            gift_name_list = random_bag.gift_name_list
            for random_gift_name,random_gift_weight in zip(gift_name_list,gift_weight_list):
                if commonutils.generate_random_number() <= mutation_rate:
                    temp_random_new_weight = commonutils.get_gift_weight(random_gift_name)
                    random_bag.remove_gift_item(random_gift_name,random_gift_weight)
                    random_bag.add_gift_item(random_gift_name,temp_random_new_weight)


def crossover_solutions(solution_list,crossover_rate=constants.cross_over_probability):
    if commonutils.generate_random_number() < crossover_rate:
        first_solution, second_solution = select_individuals(solution_list)
        index_of_swap = int((commonutils.generate_random_number()*10) % 5)
        first_bag_list = first_solution.bag_list
        second_bag_list = second_solution.bag_list
        temp_first_bag_appendage = copy.deepcopy(first_bag_list[index_of_swap:])
        temp_second_bag_appendage = copy.deepcopy(second_bag_list[index_of_swap:])
        first_solution.remove_bag_list(first_bag_list[index_of_swap:])
        first_solution.add_bag_list(temp_second_bag_appendage)
        second_solution.remove_bag_list(second_bag_list[index_of_swap:])
        second_solution.add_bag_list(temp_first_bag_appendage)
        first_solution.calculate_solution_value()


def select_individuals(solution_list):
    fitness_dict = dict({})
    for index,random_solution in enumerate(solution_list):
        random_solution.calculate_solution_value()
        fitness_dict[index] = random_solution.total_solution_value
    top_n_solution_list = commonutils.get_top_n_dict_keys(fitness_dict,2)
    first_solution = solution_list[top_n_solution_list[0][0]]
    second_solution = solution_list[top_n_solution_list[0][0]]
    return first_solution,second_solution


def genetic_solution_pipeline():
    solution_list = commonutils.generate_initial_population()
    for i in range(constants.max_iterations):
        crossover_solutions(solution_list)
        mutate_solutions(solution_list)
    best_fit_solution, _ = select_individuals(solution_list)
    best_fit_solution.calculate_solution_value()
    return best_fit_solution.total_solution_value


genetic_solution_pipeline()

