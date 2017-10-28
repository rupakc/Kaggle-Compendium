import numpy as np
import pandas as pd
import constants
import bag
import operator
np.random.seed(42)


def generate_random_numbers_range(start_range,stop_range,batch_size=1000):
    return list(np.random.randint(start_range,stop_range,size=batch_size))


def randomly_shuffle_dataframe(df):
    shuffled_data_frame = df.iloc[np.random.permutation(len(df))]
    return shuffled_data_frame


def generate_random_number():
    return np.random.random_sample()


def get_gift_weight(gift_name):
    if gift_name.lower() == 'horse':
        return max(0, np.random.normal(5,2,1)[0])
    elif gift_name.lower() == 'ball':
        return max(0, 1 + np.random.normal(1,0.3,1)[0])
    elif gift_name.lower() == 'bike':
        return max(0, np.random.normal(20,10,1)[0])
    elif gift_name.lower() == 'train':
        return max(0, np.random.normal(10,5,1)[0])
    elif gift_name.lower() == 'coal':
        return 47 * np.random.beta(0.5,0.5,1)[0]
    elif gift_name.lower() == 'book':
        return np.random.chisquare(2,1)[0]
    elif gift_name.lower() == 'doll':
        return np.random.gamma(5,1,1)[0]
    elif gift_name.lower() == 'block':
        return np.random.triangular(5,10,20,1)[0]
    elif gift_name.lower() == 'gloves':
        return 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]
    return 1.0


def get_gift_weight_helper(gift_name_raw):
    index = gift_name_raw.find('_')
    if index != -1:
        gift_weight = get_gift_weight(gift_name_raw[:index].strip().lower())
        return gift_weight
    return 3.14


def extract_gift_name(gift_name,separator='_'):
    index_of_separator = gift_name.find(separator)
    if index_of_separator != -1:
        return gift_name[:index_of_separator].strip().lower()
    return gift_name


def get_gift_weight_frame(filename='gifts.csv'):
    gift_frame = pd.read_csv(filename)
    gift_frame['weight'] = map(get_gift_weight_helper,gift_frame['GiftId'].values)
    gift_frame['GiftId'] = map(extract_gift_name,gift_frame['GiftId'].values)
    return gift_frame


def generate_mask_array(length_of_data,data_to_fill=False):
    occupied_array = [data_to_fill for i in range(length_of_data)]
    return occupied_array


def generate_initial_population(population_size=constants.initial_population_size):
    gift_frame = get_gift_weight_frame()
    gift_name_list,gift_weight_list = list(gift_frame['GiftId'].values), list(gift_frame['weight'].values)
    population_solution_list = list([])
    for i in range(population_size):
        total_population_weight = 0.0
        random_solution = bag.RandomSolution()
        taken_array = generate_mask_array(len(gift_frame))
        max_try_counter = 0
        while (max_try_counter < constants.max_tries) and (total_population_weight < (constants.max_bag_capacity*constants.max_number_of_bags-1000.0)):
            random_bag = bag.Bag()
            random_gifts = generate_random_numbers_range(0,len(gift_frame),batch_size=2000)
            max_try_counter += 1
            for gift_index in random_gifts:
                if not taken_array[gift_index]:
                    random_gift_name, random_gift_weight = gift_name_list[gift_index], gift_weight_list[gift_index]
                    random_bag.calculate_total_weight()
                    if random_bag.total_weight >= constants.max_bag_capacity-1.0:
                        break
                    if random_bag.total_weight + random_gift_weight < constants.max_bag_capacity:
                        random_bag.add_gift_item(random_gift_name,random_gift_weight)
                        taken_array[gift_index] = True
            random_solution.add_bag(random_bag)
            random_solution.calculate_solution_value()
            total_population_weight = random_solution.total_solution_value
        population_solution_list.append(random_solution)
    return population_solution_list


def get_top_n_dict_keys(dictionary,top_n=1):
    sorted_x = sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)
    sorted_x = sorted_x[:top_n]
    return sorted_x
