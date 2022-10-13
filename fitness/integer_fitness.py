import numpy as np

def f1(gene_given, gene_original):
    """
    first fitness function
    if both genes were the same, then fitness is 1 else 0

    Parameters:
    -------------
    gene_given : string
        a string of zeros and ones representing chromosomes of the given gene
    gene_original : string
        a string of zeros and ones representing chromosomes of the original gene

    Output:
    --------
    fitness : int
        the fitness of the given gene w.r.t. the original gene
    """
    fitness = None
    
    if gene_given == gene_original:
        fitness = 1
    else:
        fitness = 0
    
    return fitness

def f2(gene_given, gene_original):
    """
    second fitness function
    character-wise comparison, count the same ones

    Parameters:
    -------------
    gene_given : string
        a string of zeros and ones representing chromosomes of the given gene
    gene_original : string
        a string of zeros and ones representing chromosomes of the original gene

    Output:
    --------
    fitness : int
        the fitness of the given gene w.r.t. the original gene
    """

    ## extract the characters
    gene_given_chromosome = list(gene_given)
    gene_original_chromosomes = list(gene_original) 

    fitness = sum( np.array(gene_given_chromosome) == gene_original_chromosomes )

    return fitness

def f3(gene_given, gene_original):
    """
    third fitness function
    negative absolute distance of chromosomes in gene 

    Parameters:
    -------------
    gene_given : string
        a string of zeros and ones representing chromosomes of the given gene
    gene_original : string
        a string of zeros and ones representing chromosomes of the original gene

    Output:
    --------
    fitness : int
        the fitness of the given gene w.r.t. the original gene
    """

    ## extract the characters
    gene_given_chromosome = list(gene_given)
    gene_original_chromosomes = list(gene_original) 

    subtraction = np.array(gene_given_chromosome, dtype=int) - np.array(gene_original_chromosomes, dtype=int)

    fitness = sum(abs(subtraction))

    return fitness