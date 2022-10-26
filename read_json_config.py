from __init__ import JSON_FILE_NAME
import json
from operations.selection import binary_tournament, roulette_wheel
from operations.recombination.binary_recombination import uniform, single_point
from operations.mutation.binary_mutation import bit_flipping
from operations.mutation.integer_mutation import creep
from fitness.binary_fitness import fitness
from fitness.integer_fitness import integer_fitness


run_order_dict = None
with open(JSON_FILE_NAME) as json_file:
    run_order_dict = json.load(json_file)



def select_fitness_function(fitness_function_name, gene_original=None):
    """
    Parameters:
    ------------
    fitness_function_name : string
        fitness function name in string
    gene_original : string
        must be specified if integer genes were using
    
    Returns:
    ---------
    fitness_function : function
        return the applicable fitness function
    """
    ## setting up the fitness functions
    ## binary fitness_functions
    fitness_functions = fitness()
    
    [onemax, peak, flipflop, fourpeaks, sixpeaks, trap] = [fitness_functions.onemax, 
                                                            fitness_functions.peak, 
                                                            fitness_functions.flipflop, 
                                                            fitness_functions.fourpeaks, 
                                                            fitness_functions.sixpeaks, 
                                                            fitness_functions.trap]

    ## integer fitness functions
    integer_fitness_function = integer_fitness(gene_original) 
    [f1, f2, f3] = [integer_fitness_function.f1, integer_fitness_function.f2, integer_fitness_function.f3]

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
    elif fitness_function_name == 'f1':
        return f1
    elif fitness_function_name == 'f2':
        return f2
    elif fitness_function_name == 'f3':
        return f3
    else:
        raise ValueError(f"Error! incorrect fitness function name: {fitness_function_name}!")


def create_fitness_function_dict(fitness_function_name_arr, T=None, gene_original=None):
    """
    create a dictionary for fitness functions

    Parameters:
    ------------
    fitness_function_name_arr : array_like
        fitness function names in string 
    T : int
        the hyperparameter for fourpeaks and sixpeaks fitness functions
        must be specified if binary genes were using
    gene_original : string
        must be specified if integer genes were using
    
    Returns:
    ---------
    FITNESS_FUNCTION_dict : dictionary
        return a dictionary of fitness functions
    """
    if (T is None) and (gene_original is None):
        raise ValueError("T and gene_original variables are not specified!\nOne must be specified based on using integer or binary genes!")

    FITNESS_FUNCTION_dict = {}
    for function_name in fitness_function_name_arr:
        FITNESS_FUNCTION_dict[function_name] = select_fitness_function(function_name, gene_original)

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
    elif mutation_method_name == 'creep':
        return creep
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