from numpy.random import randint, rand

"""
Binary Recombination methods for genes
"""

def single_point(string_gene1, string_gene2):
    """
    Single point cross-over to generate new genes

    Parameters:
    ------------
    string_gene1 : string
        the first parent to generate new genes
    string_gene2 : string
        the second parent to generate new genes
    
    Output:
    --------
    child1_gene : string
        a new created child 
    child2_gene : string
        a new created child 
    """
    ## the break point in gene
    break_point = randint(1, len(string_gene1) - 1 )

    child1_gene = string_gene1[:break_point]
    child1_gene += string_gene2[break_point:]

    child2_gene = string_gene2[:break_point]
    child2_gene += string_gene1[break_point:]

    return child1_gene, child2_gene

def uniform(string_gene1, string_gene2):
    """
    uniform cross-over to generate new genes

    Parameters:
    ------------
    string_gene1 : string
        the first parent to generate new genes
    string_gene2 : string
        the second parent to generate new genes
    
    Output:
    --------
    child1_gene : string
        a new created child 
    child2_gene : string
        a new created child
    """

    ## intialize childs
    child1_gene = ''
    child2_gene = ''

    ## the uniform cross-over procedure
    for idx in range(len(string_gene1)):
        parent_selection = randint(0, 1)
        
        ## if zero, the first chromosome belongs to the first child
        ## else, the second chromosome belongs to the first child
        if parent_selection == 0:
            child1_gene += string_gene1[idx]
            child2_gene += string_gene2[idx]
        else:
            child1_gene += string_gene2[idx]
            child2_gene += string_gene1[idx]
    
    return child1_gene, child2_gene

def mutation_creep(chromosome, p_m):
    """
    mutation for the whole chromosome

    randomly change one chromosomes value
    
    Parameters:
    ------------
    chromosome : string
        a string of integer values representing chromosomes of the gene
    p_m : float
        a floating value between zero and one

    Outputs:
    ---------
    mutated_gene : string
        the mutation result
    """
    mutated_gene = chromosome
    mutated_gene = mutation_d_model(mutated_gene, p_m)
    mutated_gene = mutation_transformers(mutated_gene, p_m)
    mutated_gene = mutation_ffn_layer(mutated_gene, p_m)

    return mutated_gene


def mutation_transformers(chromosome, p_m):
    """
    transformer layers configuration mutation for chromosome

    randomly change one chromosomes value
    
    Parameters:
    ------------
    chromosome : string
        a string of integer values representing chromosomes of the gene
    p_m : float
        a floating value between zero and one

    Outputs:
    ---------
    mutated_chromosome : string
        the mutation result
    """
    ## initialize with a copy
    mutated_chromosome_list = list(chromosome)

    ## mutation on the availability or non-availability of the transfomers 1, 2 and 3
    for i in range(1, 28, 9):
        mutation_value = rand()

        ## if True, then we can mutate
        mutation_state = mutation_value < p_m

        if mutation_state:
            ## the first transformer layer is being re-configured
            if i == 1:
                mutated_layer_genes = str(randint(0, 9, 9)).replace(' ', '').replace('[','').replace(']', '')
                mutated_chromosome_list[i:i+9] = mutated_layer_genes
            ## for transformer number 2 and 3 the whole layer can be removed
            ## remove the transformer layer with a probability
            elif i != 1:
                mutated_chromosome_list[i:i+9] = '000000000'

    ## convert back to string
    mutated_chromosome = ''.join(mutated_chromosome_list)

    return mutated_chromosome

def mutation_ffn_layer(chromosome, p_m):
    """
    mutate the last 3 character of the chromosome, representing the FFN layer in our network architecture

    Parameters:
    -------------
    chromosome : string
        a string of integer values representing chromosomes of the gene
    p_m : float
        a floating value between zero and one
    
    Returns:
    ----------
    mutated_chromosome : string
        a string of integer values representing chromosomes of the mutated gene
    """
    ## create a copy to make it easier
    mutated_chromosome_list = list(chromosome)

    mutation_value = rand()

    if mutation_value < p_m:
        mutated_chromosome_list[28:31] = '000'
    # else:
    #     # mutated_layer_genes = str(randint(0, 9, 3)).replace(' ', '').replace('[','').replace(']', '')
    #     mutated_chromosome[28:31] = chromosome[28:31]


    ## convert back to string
    mutated_chromosome = ''.join(mutated_chromosome_list)
    
    return mutated_chromosome

def mutation_d_model(chromosome, p_m):
    """
    mutated just the d_model gene in the chromsome

    Parameters:
    -------------
    chromosome : string
        a string of integer values representing chromosomes of the gene
    p_m : float
        a floating value between zero and one
    
    Returns:
    ----------
    mutated_chromosome : string
        a string of integer values representing chromosomes of the mutated gene    
    
    """
    mutated_chromosome_list = list(chromosome)

    mutation_value = rand()

    if mutation_value < p_m:
        mutated_chromosome_list[0] = str(randint(0, 9))
    
    ## convert back to string
    mutated_chromosome = ''.join(mutated_chromosome_list)

    return mutated_chromosome
    