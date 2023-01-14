import numpy as np
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
    
    chromosome_counts = len(chromosomes_arr)
    ## initialization
    ## the indexes of chromosomes showing it could dominate other or not
    pareto_dominance_indexes = []
    
    ## for each chromosome
    for i in range(chromosome_counts):
        ## index to exclude
        indexes = np.ones(chromosome_counts, dtype=bool)
        indexes[i] = False

        dominated = find_pareto_dominance(fitness_values_arr[i], fitness_values_arr[indexes])
        ## if False, then it is not dominated by others and can be in pareto-set
        pareto_dominance_indexes.append(not dominated)

    ## the pareto-front set
    pareto_dominant = chromosomes_arr[pareto_dominance_indexes] 
    ## fitnesses of pareto-front
    pareto_optimal = fitness_values_arr[pareto_dominance_indexes]
    
    other_chromosomes = chromosomes_arr[~ np.array(pareto_dominance_indexes)]

    other_chromosomes_fitness = fitness_values_arr[~ np.array(pareto_dominance_indexes)] 
    
    return pareto_dominant, pareto_optimal, other_chromosomes, other_chromosomes_fitness

def find_pareto_dominance(fitness_set, fitness_values_arr):
    """
    find out whether the fitness values of a chromosome is dominated by others or not
    if it was dominated True is returned, then it couldn't be in pareto_dominance set 

    Parameters:
    -------------
    fitness_set : array_like
        the fitness values of a chromosome representing multiple fitness functions 
    fitness_values_arr : array_like 
        numpy multi-dimensional array representing the fitness values of chromosomes of different fitness functions excluding the fitness_set

    Returns:
    ---------
    is_dominated : bool
        whether the fitness values of a chromosome is dominated by others or not
    """
    ## minimization
    fitness_comparison = fitness_set <= fitness_values_arr

    is_dominated = np.all(np.any(fitness_comparison, axis=1))

    return not is_dominated