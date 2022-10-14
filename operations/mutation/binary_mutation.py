"""
Mutation method for binary genes
"""

# from random import random
from numpy.random import randint


def bit_flipping(string_gene):
    """
    bit flipping mutation method for a binary gene

    randomly flip one chromosomes value
    
    Parameters:
    ------------
    string_gene : string
        a string of zeros and ones representing chromosomes of the gene

    Outputs:
    ---------
    mutated_gene : string
        the mutation result
    """
    ## initialize empty gene
    mutated_gene = ''

    ## randomly select a gene in chromosome to be flipped
    chromosome_idx = randint(0, len(string_gene))

    not_chromosome = '0' if string_gene[chromosome_idx] == '1' else '1'
    
    return mutated_gene



    