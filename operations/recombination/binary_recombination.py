from random import randint

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


