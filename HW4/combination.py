"""
Mutation and Cross-over are applied here

"""

import random
from generate_population_scripts import sampling_function, process_division_points, divide_vehicles
import numpy as np


def split_string_step(string, step):
    """
    split a string based on steps
    """
    string_arr = []

    for i in range(1, len(string)+1):
        if i % step == 0:
            string_arr.append(string[i-step:i])
    return string_arr

def remove_symbols_chromsome(chromsome, depot_symbols_arr):
    """
    remove all the division and depot symbols in the chromsome
    """
    raw_chromsome = chromsome.replace('|', '')

    for symbol in depot_symbols_arr:
        raw_chromsome = raw_chromsome.replace(symbol, '')

    return raw_chromsome

def process_depots_capacity_based(dataset, string_chromsome_arr, max_capacity, depot_symbol):
    """
    Add the depot symbols to string chromsome array (the customer numbers are strings in the array) based on capacity constraint
    """
    capacity = 0
    idx = 0
    string_chromsome = depot_symbol
    while idx < len(string_chromsome_arr):
        gene = string_chromsome_arr[idx]

        customer_no = int(gene) - 100
        customer = dataset[dataset.number == customer_no]

        capacity += customer.demand.values[0]
        
        if capacity > max_capacity:
            string_chromsome += depot_symbol
            capacity = 0
        else:
            string_chromsome += gene
            idx += 1

    string_chromsome += depot_symbol if string_chromsome[-3:] != depot_symbol else ''
    
    
    return string_chromsome
def process_depots_distance_based(dataset, string_chromsome_arr, max_distance, depot_symbol, depot_location):
    """
    Add the depot symbols to string chromsome array (the customer numbers are strings in the array) based on distance constraint
    """
    distance = 0
    idx = 0
    string_chromsome = depot_symbol
    last_X, last_Y = depot_location
    while idx < len(string_chromsome_arr):
        gene = string_chromsome_arr[idx]
        customer_no = int(gene) - 100
        customer = dataset[dataset.number == customer_no]

        customer_X = customer.x.values[0]
        customer_Y = customer.y.values[0]
        
        distance_to_depot = distance + abs(last_X - depot_location[0]) + abs(last_Y - depot_location[1])

        ## manhatan distance
        distance += abs(last_X - customer_X) + abs(last_Y - customer_Y)

        # print(distance, distance_to_depot, max_distance, customer_no, customer_X, customer_Y)
        
        if distance < max_distance:
            string_chromsome += gene

            ## update the last locations
            last_X, last_Y = customer_X, customer_Y
            idx += 1
        ## if by going to another customer would overcome our limit
        ## and going back to depot is possible (less than max distance)
        elif distance >= max_distance and distance_to_depot <= max_distance:
            string_chromsome += depot_symbol
            last_X, last_Y = depot_location
            distance = 0
        ## if we also couldn't afford to go back to the depot
        else:
            ## the last customer's length is measured
            last_customer_length = len(string_chromsome_arr[idx])

            ## if the last one was depot then, don't decrease the index or remove the gene
            if string_chromsome[-last_customer_length:] != depot_symbol:
                distance = 0
                continue

            ## remove the last customer
            string_chromsome = string_chromsome[:-last_customer_length]
            ## and go back to the depot instead of the previous customer
            string_chromsome += depot_symbol
            distance = 0
            
            last_X, last_Y = depot_location
            ## decrease the index
            idx -= 1
    
    return string_chromsome

def get_chromosome_break_point(chromosome=None, step=3, max_index=None):
    """
    generate randomly a break point for a given chromsome
    Parameters:
    -------------
    chromsome : string
        a raw chromsome without any symbols of depot or vehicle division symbols
    step : integer
        defines the interval of genes which belong to one phenotype
    max_index : integer
        represents the maximum number can be reached, if None it would be the length of chromsome minus step
    
    Note: one of the `chromosome` or the `max_index` must be given as the input, if both were None raise an error! 
    (if both given, then the max_index has the priority and would be used)

    Returns:
    ---------
    break_point : int
        a breaking point for chromsome 
    """
    ## generate an array to randomly sample from
    ## the break point should not be the first or the last points
    ## create an array and randomly sample from it
    if max_index is None and chromosome is not None:
        arr = np.arange(0, len(chromosome) +1, step)
        ## if the produced array is empty
        ## we manually add an element to it (we know the break point will be same as step!)
        if len(arr) == 0:
            arr = [step]

    elif max_index is not None:
        arr = np.arange(0, max_index , step)
    else:
        raise ValueError("Both chromsome and max_index should not be None!")
    
    ## randomly sampling
    break_point = sampling_function(list(arr))
    break_point = break_point // step

    return break_point

def repair_chromsome(raw_chromsome, dataset, max_capacity, depot_location_dict, vehicle_count, max_distance, vehicle_depot_constraint):
    """
    repair the chromsome by adding the depot symbol and division points for the vehicles
    """
    depot_symbol = random.choice(list(depot_location_dict.keys()))
    depot_location = depot_location_dict[depot_symbol]

    chromosome = None
    if max_distance is None:
        chromosome = process_depots_capacity_based(dataset, raw_chromsome, max_capacity, depot_symbol)
        chromosome = divide_vehicles(chromosome, vehicle_count, depot_location_dict, max_distance_constraint= (max_distance is not None), vehicle_depot_constraint = vehicle_depot_constraint )
    else:
        chromosome = process_depots_distance_based(dataset, raw_chromsome, max_distance, depot_symbol, depot_location)
        chromosome = divide_vehicles(chromosome, vehicle_count, depot_location_dict, max_distance_constraint= True, vehicle_depot_constraint = vehicle_depot_constraint )

    # chromosome = process_division_points(chromosome, depot_symbol, vehicle_depot_constraint)
    chromosome = process_division_points(chromosome, list(depot_location_dict.keys()), vehicle_depot_constraint)



    return chromosome


def cut_and_crossfill(chromosome1, chromosome2, dataset, max_capacity, max_distance, depot_location_dict, vehicle_depot_constraint, vehicle_count):
    """
    cut and crossfill cross-over

    at the end repair the chromsome with max_capacity
    """

    ## find the chromsome with minimum length and break it
    chromosome_to_break = chromosome1 if len(chromosome1) < len(chromosome2) else chromosome2
    second_chromsome = chromosome1 if len(chromosome1) > len(chromosome2) else chromosome2
    
    chromosome = remove_symbols_chromsome(chromosome_to_break, list(depot_location_dict.keys()))
    second_chromsome = remove_symbols_chromsome(second_chromsome, list(depot_location_dict.keys()))
    
    ## break point of the chromsomes
    break_point = get_chromosome_break_point(chromosome=None, step=3, max_index=min(len(chromosome), len(second_chromsome)) - 3)
    
    chromosome_arr = split_string_step(chromosome, 3)
    second_chromsome_arr = split_string_step(second_chromsome, 3)

    ## offspring creation
    offspring1 = chromosome_arr[:break_point]

    for gene in second_chromsome_arr[break_point:] + second_chromsome_arr[:break_point]:
        if gene not in offspring1:
            offspring1.append(gene)
    
    offspring2 = second_chromsome_arr[break_point:]
    for gene in chromosome_arr[:break_point] + chromosome_arr[break_point:]:
        if gene not in offspring2:
            offspring2.append(gene)

    ## repairment
    offspring1 = repair_chromsome(offspring1, dataset, max_capacity, depot_location_dict, vehicle_count, max_distance, vehicle_depot_constraint)

    offspring2 = repair_chromsome(offspring2, dataset, max_capacity, depot_location_dict, vehicle_count, max_distance, vehicle_depot_constraint)


    return offspring1, offspring2
def find_two_mutation_point(chromosome):
    """
    find two mutation points and return those
    """
    mutation_point1 = get_chromosome_break_point(chromosome)
    mutation_point2 = mutation_point1
    ## go until it finds another index
    while mutation_point2 == mutation_point1:
        mutation_point2 = get_chromosome_break_point(chromosome)
    
    ## sort mutation points    
    mutation_point1, mutation_point2 = np.sort((mutation_point1, mutation_point2))

    return mutation_point1, mutation_point2

def mutation_inverse(chromosome, max_capacity, dataset, depot_location_dict, max_distance, vehicle_depot_constraint, vehicle_count):
    """
    inverse mutation for the genes of a chromsome, returns the mutated chromsome
    """

    raw_chromsome = remove_symbols_chromsome(chromosome, depot_location_dict)
    
    mutation_point1, mutation_point2 = find_two_mutation_point(raw_chromsome)

    raw_chromsome_arr = split_string_step(raw_chromsome, 3)

    inversed_chromosome_arr = raw_chromsome_arr[:mutation_point1] + raw_chromsome_arr[mutation_point1:mutation_point2][::-1] + raw_chromsome_arr[mutation_point2:]

    repaired_inversed_chromosome = repair_chromsome(inversed_chromosome_arr, dataset, max_capacity, depot_location_dict, vehicle_count, max_distance, vehicle_depot_constraint)


    return repaired_inversed_chromosome

def mutation_scramble(chromosome, max_capacity, dataset, depot_location_dict, max_distance, vehicle_depot_constraint, vehicle_count):
    """
    scramble mutation for the genes of a chromsome, returns the mutated chromsome
    """
    raw_chromsome = remove_symbols_chromsome(chromosome, list(depot_location_dict.keys()))

    mutation_point1, mutation_point2 = find_two_mutation_point(raw_chromsome)

    raw_chromsome_arr = split_string_step(raw_chromsome, 3)

    ## the customers number to be scrambled
    to_scramble_arr = list(raw_chromsome_arr[mutation_point1:mutation_point2])

    chromsome_scrambled_arr = raw_chromsome_arr[:mutation_point1]

    while len(to_scramble_arr) != 0:
        customer = sampling_function(to_scramble_arr)
        chromsome_scrambled_arr += [customer]
    
    chromsome_scrambled_arr += raw_chromsome_arr[mutation_point2:]

    mutated_chromosome = repair_chromsome(chromsome_scrambled_arr, dataset, max_capacity, depot_location_dict, vehicle_count, max_distance, vehicle_depot_constraint)


    return mutated_chromosome