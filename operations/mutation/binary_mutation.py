"""
Mutation method for binary genes
"""

from random import random


def bit_flipping(string_gene, p_m):
    """
    bit flipping mutation method for a binary gene

    randomly flip one chromosomes value
    
    Parameters:
    ------------
    string_gene : string
        a string of zeros and ones representing chromosomes of the gene
    p_m : float between zero an one
        the probability of bit flipping mutation

    Outputs:
    ---------
    mutated_gene : string
        the mutation result
    """
    ## initialize empty gene
    mutated_gene = ''

    for chromosome in string_gene:
        ## generate a random value to see if we want to mutate or not
        mutation_value = random()
        
        ## if True, then we can mutate
        mutation_state = mutation_value < p_m

        not_chromosome = '0' if chromosome == '1' else '1'

        ## add to the new gene
        mutated_gene += not_chromosome if mutation_state else chromosome
    
    return mutated_gene



    