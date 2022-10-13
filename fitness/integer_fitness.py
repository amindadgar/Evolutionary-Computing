import numpy as np

class integer_fitness():

    def __init__(self, gene_original) -> None:
        """
        Parameters:
        -------------
        gene_original : string
            a string of zeros and ones representing chromosomes of the original gene
        """
        self.gene_original = gene_original
        self.gene_original_chromosomes = list(self.gene_original) 


    def f1(self, gene_given) -> int:
        """
        first fitness function
        if both genes were the same, then fitness is 1 else 0

        Parameters:
        -------------
        gene_given : string
            a string of zeros and ones representing chromosomes of the given gene


        Output:
        --------
        fitness : int
            the fitness of the given gene w.r.t. the original gene
        """
        fitness = None
        
        if gene_given == self.gene_original:
            fitness = 1
        else:
            fitness = 0
        
        return fitness

    def f2(self, gene_given) -> int:
        """
        second fitness function
        character-wise comparison, count the same ones

        Parameters:
        -------------
        gene_given : string
            a string of zeros and ones representing chromosomes of the given gene

        Output:
        --------
        fitness : int
            the fitness of the given gene w.r.t. the original gene
        """

        ## extract the characters
        gene_given_chromosome = list(gene_given)

        fitness = sum( np.array(gene_given_chromosome) == self.gene_original_chromosomes )

        return fitness

    def f3(self, gene_given) -> int:
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

        subtraction = np.array(gene_given_chromosome, dtype=int) - np.array(self.gene_original_chromosomes, dtype=int)

        fitness = sum(abs(subtraction))

        return fitness