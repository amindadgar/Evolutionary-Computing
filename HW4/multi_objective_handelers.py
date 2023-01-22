import numpy as np
from paretoset import paretoset
"""
handle multi-objective side of a problem
"""

def find_pareto_set(chromosomes_arr, fitness_values_arr):
    """
    find pareto set based on fitness values

    Parameters:
    ------------
    chromosomes_arr : array_like
        array of chromosomes representing the solutions in problem
    fitness_values_arr : array_like
        numpy multi-dimensional array representing the fitness values of chromosomes of different fitness functions

    Returns:
    ----------
    pareto_dominant : array_like
        An array of chromosomes which are in pareto-front set
    pareto_optimal : array_like
        An array of fitnesses of chromosomes which are in pareto-front set
    other_chromosomes : array_like
        other chromosomes not in pareto-front
    other_chromosomes_fitness : array_like
        fitness of other chromosomes not in pareto-front
    """
    

    pareto_dominance_indexes = paretoset(fitness_values_arr, sense=['min', 'max'])

    ## the pareto-front set
    pareto_dominant = chromosomes_arr[pareto_dominance_indexes] 
    ## fitnesses of pareto-front
    pareto_optimal = fitness_values_arr[pareto_dominance_indexes]
    
    other_chromosomes = chromosomes_arr[~ np.array(pareto_dominance_indexes)]

    other_chromosomes_fitness = fitness_values_arr[~ np.array(pareto_dominance_indexes)] 
    
    return pareto_dominant, pareto_optimal, other_chromosomes, other_chromosomes_fitness
