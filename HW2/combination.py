"""
Mutation and Cross-over are applied here

"""

import random
from generate_population_scripts import without_replacement_sampling, process_division_points, divide_vehicles
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

def remove_symbols_chromsome(chromsome, depot_symbols):
    """
    remove all the division and depot symbols in the chromsome
    """
    raw_chromsome = chromsome.replace('|', '')

    for symbol in depot_symbols:
        raw_chromsome = raw_chromsome.replace(symbol, '')

    return raw_chromsome

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
        arr = np.arange(0 + step, len(chromosome) - step , step)
    elif max_index is not None:
        arr = np.arange(0 + step, max_index , step)
    else:
        raise ValueError("Both chromsome and max_index should not be None!")
    
    ## randomly sampling
    break_point = without_replacement_sampling(list(arr))
    break_point = break_point // step

    return break_point

def repair_chromsome(raw_chromsome, dataset, max_capacity, depot, vehicle_count):
    """
    repair the chromsome by adding the depot symbol and division points for the vehicles
    """

    chromosome = process_depots(dataset, raw_chromsome, max_capacity, depot)
    chromosome = divide_vehicles(chromosome, vehicle_count, depot)
    chromosome = process_division_points(chromosome, depot)

    return chromosome


def cut_and_crossfil(chromosome1, chromosome2, dataset, max_capacity, depot_symbol=['(1)']):
    """
    cut and crossfill cross-over

    at the end repair the chromsome with max_capacity
    """
    ## find the vehicle count
    vehicle_count = chromosome1.count('|') + 1

    ## find the chromsome with minimum length and break it
    chromosome_to_break = chromosome1 if len(chromosome1) < len(chromosome2) else chromosome2
    second_chromsome = chromosome1 if len(chromosome1) > len(chromosome2) else chromosome2
    
    chromosome = remove_symbols_chromsome(chromosome_to_break, depot_symbol)
    second_chromsome = remove_symbols_chromsome(second_chromsome, depot_symbol)
    
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
    random_depot = random.sample(depot_symbol, 1)[0]
    offspring1 = repair_chromsome(offspring1, dataset, max_capacity, random_depot, vehicle_count)

    random_depot = random.sample(depot_symbol, 1)[0]
    offspring2 = repair_chromsome(offspring2, dataset, max_capacity, random_depot, vehicle_count)


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

def mutation_inverse(chromosome, max_capacity, dataset, depot_symbol=['(1)']):
    """
    inverse mutation for the genes of a chromsome, returns the mutated chromsome
    """
    vehicle_count = chromosome.count('|') + 1
    raw_chromsome = remove_symbols_chromsome(chromosome, depot_symbol)
    
    mutation_point1, mutation_point2 = find_two_mutation_point(raw_chromsome)

    raw_chromsome_arr = split_string_step(raw_chromsome, 3)

    inversed_chromosome_arr = raw_chromsome_arr[:mutation_point1] + raw_chromsome_arr[mutation_point1:mutation_point2][::-1] + raw_chromsome_arr[mutation_point2:]

    random_depot = random.sample(depot_symbol, 1)[0]
    repaired_inversed_chromosome = repair_chromsome(inversed_chromosome_arr, dataset, max_capacity, random_depot, vehicle_count)

    return repaired_inversed_chromosome

def mutation_scramble(chromosome, max_capacity, dataset, depot_symbol=['(1)']):
    """
    scramble mutation for the genes of a chromsome, returns the mutated chromsome
    """
    vehicle_count = chromosome.count('|') + 1
    raw_chromsome = remove_symbols_chromsome(chromosome, depot_symbol)

    mutation_point1, mutation_point2 = find_two_mutation_point(raw_chromsome)

    raw_chromsome_arr = split_string_step(raw_chromsome, 3)

    ## the customers number to be scrambled
    to_scramble_arr = list(raw_chromsome_arr[mutation_point1:mutation_point2])

    chromsome_scrambled_arr = raw_chromsome_arr[:mutation_point1]

    while len(to_scramble_arr) != 0:
        customer = without_replacement_sampling(to_scramble_arr)
        chromsome_scrambled_arr += [customer]
    
    chromsome_scrambled_arr += raw_chromsome_arr[mutation_point2:]

    random_depot = random.sample(depot_symbol, 1)[0]
    mutated_chromosome = repair_chromsome(chromsome_scrambled_arr, dataset, max_capacity, random_depot, vehicle_count)

    return mutated_chromosome