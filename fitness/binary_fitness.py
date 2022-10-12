import numpy as np


class binary_fitness():
    "Binary fitness functions for evolutionary algorithm"

    def onemax(string_gene) -> int:
        """
        OneMax fitness function
        size of chromosomes with value `1`

        Parameters:
        ------------
        string_gene : string
            a string of zeros and ones representing chromosomes of a gene

        Output:
        ---------
        fitness : int
            the fitness of the gene
        """
        fitness = str.count(string_gene, '1')
        return fitness

    def peak(string_gene) -> int:
        """
        peak fitness function
        multiplication of the chromosome

        Parameters:
        ------------
        string_gene : string
            a string of zeros and ones representing chromosomes of a gene

        Output:
        ---------
        fitness : int
            the fitness of the gene
        """

        fitness = None
        ## if a zero was available in string, then fitness will be zero
        ## else 1
        if str.count(string_gene, '0') == 0:
            fitness = 0
        else:
            fitness = 1
        
        ## if any problems happen 
        if fitness == None:
            raise f"Error! fitness in peak function is not set!\nCheck your gene: {string_gene}" 

        
        return fitness

    

    def flipflop(string_gene) -> int:
        """
        flipflop fitness function
        xor the chromosomes with each other

        Parameters:
        ------------
        string_gene : string
            a string of zeros and ones representing chromosomes of a gene

        Output:
        ---------
        fitness : int
            the fitness of the gene
        """

        ## to avoid loop we will shift the binary gene and xor it with the shifted one

        ## shifting and addint `1` to the first chromosome
        string_gene_shifted = '1' + bin(int(string_gene, 2) >> 1)[2:]

        ## xor operation can now be applied
        genes_xor = bin(int(string_gene, 2) ^ int(string_gene_shifted, 2))[2:]

        ## count of ones in gene is the fitness
        fitness = str.count(genes_xor, '1')

        return fitness