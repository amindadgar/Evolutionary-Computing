## generate the population

from numpy.random import randint

def generate_population(pop_size=10):
    """
    generate population 
    each chromsome have 9 genes, representing the hyperparameters of the problem
    
    """

    chromsomes = []

    for i in range(pop_size):
        genes_arr = randint(0, 10, 31)

        ## process it to make string
        genes = str(genes_arr).replace(' ', '').replace('[','').replace(']', '') 

        chromsomes.append(genes)
    
    return chromsomes