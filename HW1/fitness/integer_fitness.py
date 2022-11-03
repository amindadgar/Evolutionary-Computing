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
        gene_original_sliced = self.__process_original_gene(self.gene_original, len(list(gene_given)))

        if gene_given == gene_original_sliced:
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

        gene_original_sliced = self.__process_original_gene(self.gene_original, len(list(gene_given)))
        
        ## extract the characters
        gene_given_chromosome = list(gene_given)
        gene_original_chromosomes = list(gene_original_sliced)

        fitness = sum( np.array(gene_given_chromosome) == np.array(gene_original_chromosomes) )

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
        gene_original_sliced = self.__process_original_gene(self.gene_original, len(list(gene_given)))

        ## extract the characters
        gene_given_chromosome = list(gene_given)
        gene_original_chromosomes = list(gene_original_sliced)

        subtraction = np.int16(gene_given_chromosome) - np.int16(gene_original_chromosomes)

        fitness = -1 * sum(abs(subtraction))

        return fitness

    def __process_original_gene(self, gene_original, slice_idx):
        """
        slice the original gene with the problem size length

        Parameters:
        -------------
        original_gene : string
            the original gene with high length
        slice_idx : int
            the index to slice the original gene for its length matches the gene given for fitness functions
        
        Returns:
        ---------
        gene_sliced : string
            string gene sliced with index value
        """

        return gene_original[:slice_idx]
        