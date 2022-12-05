from util import convert_genotype_to_phenotype_values

def static_fitness(chromosome):
    """
    return a static value for the fitness for debugging purposes
    """

    return 1

def fitness_training(chromosome, training_count = 5):
    """
    find the fitness of the chromosome with training the model and returning the `training_count` average test accuracies

    Parameters:
    ------------
    chromosome : string
        the string representing the architecture of the transformer network
    training_count : int
        the count of training needs for averaging the fitness value
    
    Returns:
    ---------
    chromsome_fitness : float
        the average accuracy of test set in 5 time training the network
    """

    phenotype_values = convert_genotype_to_phenotype_values(chromosome)
    ## TODO

    