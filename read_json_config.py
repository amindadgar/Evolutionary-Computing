JSON_FILE_NAME = "run_order.json"

import json
from operations.selection import binary_tournament, roulette_wheel
from operations.recombination.binary_recombination import uniform, single_point
from operations.mutation.binary_mutation import bit_flipping
from fitness.binary_fitness import fitness


run_order_dict = None
with open('run_order.json') as json_file:
    run_order_dict = json.load(json_file)



def select_fitness_function(fitness_function_name, T):
    """
    Parameters:
    ------------
    fitness_function_name : string
        fitness function name in string
    T : int
        the hyperparameter for fourpeaks and sixpeaks fitness functions
    
    Returns:
    ---------
    fitness_function : function
        return the applicable fitness function
    """
    ## setting up the fitness functions
    fitness_functions = fitness(T)
    [onemax, peak, flipflop, fourpeaks, sixpeaks, trap] = [fitness_functions.onemax, 
                                                            fitness_functions.peak, 
                                                            fitness_functions.flipflop, 
                                                            fitness_functions.fourpeaks, 
                                                            fitness_functions.sixpeaks, 
                                                            fitness_functions.trap]

    if fitness_function_name == 'onemax':
        return onemax
    elif fitness_function_name == 'peak':
        return peak
    elif fitness_function_name == 'flipflop':
        return flipflop
    elif fitness_function_name == 'fourpeaks':
        return fourpeaks
    elif fitness_function_name == 'sixpeaks':
        return sixpeaks
    elif fitness_function_name == 'trap':
        return trap
    else:
        raise ValueError(f"Error! incorrect fitness function name: {fitness_function_name}!")


def create_fitness_function_dict(fitness_function_name_arr, T):
    """
    create a dictionary for fitness functions
    """
    FITNESS_FUNCTION_dict = {}
    for function_name in fitness_function_name_arr:
        FITNESS_FUNCTION_dict[function_name] = select_fitness_function(function_name, T)

    return FITNESS_FUNCTION_dict

def select_selection_method_function(selection_method_name):
    """
    select the selection method function by the name of it
    """

    if selection_method_name == 'roulette_wheel':
        return roulette_wheel
    elif selection_method_name == 'binary_tournament':
        return binary_tournament
    else:
        
        raise ValueError(f"Error! incorrect selection method name: {selection_method_name}!")

def create_selection_method_dict(selection_method_arr):
    """
    creaet a dictionary for selection methods 
    """

    SELECTION_METHOD_dict = {}
    for selection_name in selection_method_arr:
        SELECTION_METHOD_dict[selection_name] = select_selection_method_function(selection_name)
    
    return SELECTION_METHOD_dict


def select_recombination_method_function(recombination_method_name):
    """
    select the recombination method function by the name of it
    """

    if recombination_method_name == 'uniform':
        return uniform
    elif recombination_method_name == 'single_point':
        return single_point
    else:
        raise ValueError(f"Error! incorrect recombination method name: {recombination_method_name}!")

def create_recombination_method_dict(recombination_method_arr):
    """
    creaet a dictionary for recombination methods 
    """

    RECOMBINATION_METHOD_dict = {}
    for recombination_name in recombination_method_arr:
        RECOMBINATION_METHOD_dict[recombination_name] = select_recombination_method_function(recombination_name)
    
    return RECOMBINATION_METHOD_dict

def select_mutation_method_function(mutation_method_name):
    """
    select the mutation method function by the name of it
    """

    if mutation_method_name == 'bit_flipping':
        return bit_flipping
    else:
        raise ValueError(f"Error! incorrect recombination method name: {mutation_method_name}!")

def create_mutation_method_dict(mutation_method_arr):
    """
    creaet a dictionary for mutation methods 
    """

    MUTATION_METHOD_dict = {}
    for mutation_name in mutation_method_arr:
        MUTATION_METHOD_dict[mutation_name] = select_mutation_method_function(mutation_name)
    
    return MUTATION_METHOD_dict