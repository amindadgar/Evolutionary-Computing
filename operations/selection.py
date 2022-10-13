from numpy import random
"""
Selection methods 
"""

def roulette_wheel(population, selection_pop_size, fitness_function, with_replacement = True) -> list:
    """
    roulette wheel (fitness proportionate) Method to select the new population
    

    Parameters:
    ------------
    population : array
        array of genes
        each gene corresponds to one person
    fitness_function : function
        the function that calculates the fitness
        note that the input for the function should be a string gene (one string input)
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

    ## intialize the population fitness array
    fitness_pop = []
    for gene in population:
        fitness_gene = fitness_function(gene)

        fitness_pop.append(fitness_gene)

    population_probability = fitness_pop / sum(fitness_pop)

    selected_population = random.choice(population, 
                            size=selection_pop_size,
                            p=population_probability,
                            raplce=with_replacement)

    return selected_population


def binary_tournament(population, selection_pop_size, fitness_function, with_replacement = True) -> list:
    """
    binary tournament Method to select the new population
    
    Parameters:
    ------------
    population : array
        array of genes
        each gene corresponds to one person
    fitness_function : function
        the function that calculates the fitness
        note that the input for the function should be a string gene (one string input)
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
    ## initialize an empty array
    selected_population = []

    ## the gene selection
    for _ in range(selection_pop_size):
        ## get out two gene from population randomly (uniform random)
        gene1, gene2 = random.choice(population, size=2)

        ## evaluate which has a better fitness
        if fitness_function(gene1) > fitness_function(gene2):
            selected_population.append(gene1)
        else:
            selected_population.append(gene2)
    
    return selected_population