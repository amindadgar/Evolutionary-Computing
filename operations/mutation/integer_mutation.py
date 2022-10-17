"""
Mutation method for integer genes
"""

from numpy.random import rand, randint


def creep(string_gene, p_m):
    """
    creep mutation method for an integer gene

    randomly change one chromosomes value
    
    Parameters:
    ------------
    string_gene : string
        a string of zeros and ones representing chromosomes of the gene
    p_m : float
        a floating value between zero and one

    Outputs:
    ---------
    mutated_gene : string
        the mutation result
    """
    ## initialize empty gene
    mutated_gene = ''

    for chromosome in string_gene:
        ## generate a random value to see if we want to mutate or not
        mutation_value = rand()
        
        ## if True, then we can mutate
        mutation_state = mutation_value < p_m

        ## add to the new gene
        ## if we had to mutate
        if mutation_state:
            ## for integers we randomly choose a value for mutation
            mutated_chromosome = randint(low=0, high=10)
            mutated_gene += mutated_chromosome

        ## else add the same chromosome
        else:
            mutated_chromosome += chromosome
    
    return mutated_gene



    