from numpy import random, where
"""
Selection methods 
"""

def roulette_wheel(population, selection_pop_size, pop_fitness, with_replacement = True) -> list:
    """
    roulette wheel (fitness proportionate) Method to select the new population
    

    Parameters:
    ------------
    population : array
        array of genes
        each gene corresponds to one person
    pop_fitness : array
        the fitness of population
        each item of array indicates each gene fitness
    selction_pop_size : int
        specify how many genes to return
    with_replacement : bool
        select genes from population with replacement or not
        default is True, meaning a gene can be chosen multiple times 
    
    Output:
    --------
    selected_population : array
        the selected population for the new generation
    """

    # find the probability of selection of each gene in population
    population_probability = pop_fitness / sum(pop_fitness)

    # new population selection
    selected_population = random.choice(population, 
                            size=selection_pop_size,
                            p=population_probability,
                            raplce=with_replacement)

    return selected_population


def binary_tournament(population, selection_pop_size, pop_fitness) -> list:
    """
    binary tournament Method to select the new population
    
    Parameters:
    ------------
    population : array
        array of genes
        each gene corresponds to one person
    pop_fitness : array
        the fitness of population
        each item of array indicates each gene fitness
    selction_pop_size : int
        specify how many genes to return
    
    Output:
    --------
    selected_population : array
        the selected population for the new generation
    """
    # initialize an empty array
    selected_population = []

    # the gene selection
    for _ in range(selection_pop_size):
        # get out two gene from population randomly (uniform random)
        gene1, gene2 = random.choice(population, size=2)
        
        ## find the indexes in order to get the fitness from pop_fitness array
        gene1_idx = where(population == gene1)
        gene2_idx = where(population == gene2)

        # evaluate which has a better fitness
        if pop_fitness[gene1_idx] > pop_fitness[gene2_idx]:
            selected_population.append(gene1)
        else:
            selected_population.append(gene2)
    
    return selected_population