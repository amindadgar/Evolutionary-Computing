import numpy as np


class binary_fitness():
    "Binary fitness functions for evolutionary algorithm"


    def __init__(self) -> None:
        pass

    def onemax(self, string_gene) -> int:
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

    def peak(self, string_gene) -> int:
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
        if str.count(string_gene, '0') > 0:
            fitness = 0
        else:
            fitness = 1
        
        ## if any problems happen 
        if fitness == None:
            raise f"Error! fitness in peak function is not set!\nCheck your gene: {string_gene}" 

        
        return fitness

    

    def flipflop(self, string_gene) -> int:
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

        ## shifting and adding the first chromosome of the gene to have the same length
        ## same as logical right shift
        string_gene_shifted = string_gene[0] + string_gene[:-1]

        ## xor operation can now be applied
        genes_xor = bin(int(string_gene, 2) ^ int(string_gene_shifted, 2))[2:]

        ## count of ones in gene is the fitness
        fitness = str.count(genes_xor, '1')

        return fitness


    def fourpeaks(self, string_gene, T) -> int:
        """
        fourpeaks fitness function
        equation is
        `max(tail(0, string_gene), head(1, string_gene)) + R`
        where `R` is `gene_length` if both `tail(0, string_gene)` and `head(1, string_gene)` are bigger than an integer `T` else is `0` 
        `tail` and `head` are functions that count the zeros in the tail and ones in the head of the gene respectively 

        Parameters:
        ------------
        string_gene : string
            a string of zeros and ones representing chromosomes of a gene
        T : integer
            hyperparameter for the `R` function

        Output:
        ---------
        fitness : int
            the fitness of the gene
        """

        fitness = max(self.__tail(string_gene, '0'), self.__head(string_gene, '1')) + self.__R(string_gene, T, 'fourpeak')

        return fitness

    def sixpeaks(self, string_gene, T) -> int:
        """
        sixpeaks fitness function
        equation is
        `max(tail(0, string_gene), head(1, string_gene)) + R`
        where `R` is `gene_length` if both `tail(0 or 1, string_gene)` and `head(1 or 0, string_gene)` are bigger than an integer `T` else is `0` 
        `tail` and `head` are functions that count the zeros in the tail and ones in the head of the gene respectively 

        Parameters:
        ------------
        string_gene : string
            a string of zeros and ones representing chromosomes of a gene
        T : integer
            hyperparameter for the `R` function

        Output:
        ---------
        fitness : int
            the fitness of the gene
        """

        fitness = max(self.__tail(string_gene, '0'), self.__head(string_gene, '1')) + self.__R(string_gene, T, 'sixpeak')

        return fitness

    def trap(self, string_gene):
        """
        Trap fitness function
        equation is
        `3 * gene_length * peak(string_gene) - onemax(string_gene)` 
        
        Parameters:
        ------------
        string_gene : string
            a string of zeros and ones representing chromosomes of a gene
            
        Output:
        ---------
        fitness : int
            the fitness of the gene
        """
        fitness = 3 * len(string_gene) * self.peak(string_gene) - self.onemax(string_gene)

        return fitness


    def __R(self, string_gene, T, fitness_function):
        """
        find the value for the `R` function in for either sixpeak or fourpeak fitness functions

        Parameters:
        ------------
        string_gene : string
            a string of zeros and ones representing chromosomes of a gene
        T : integer
            hyperparameter for the `R` function
        fitness_function : string
            is either 'sixpeak' or 'fourpeak'

        Output:
        ---------
        R_res : int
            the result function
        """

        if fitness_function == 'sixpeak':
            ## because the conditions were too lengthy, we seperated them in two variables
            condition1 = (self.__tail(string_gene, '0') > T) or self.__head(string_gene, '1') > T
            condition2 = (self.__tail(string_gene, '1') > T) or self.__head(string_gene, '0') > T

            condition = condition1 or condition2

        elif fitness_function == 'fourpeak':
            condition = (self.__tail(string_gene, '0') > T) or self.__head(string_gene, '1') > T
        else:
            raise ValueError(f'fitness_function variable should be either \'sixpeak\' or \'fourpeak\'\nNow is {fitness_function}')


        if condition:
            R_res = len(string_gene)
        else:
            R_res = 0

        return R_res
        


    def __tail(self, string_gene, value):
        """
        find count of values with `value` in gene in the tail of it

        Parameters:
        ------------
        string_gene : string
            a string of zeros and ones representing chromosomes of a gene
        T : integer
            value of chromosome
            shoud be either `0` or `1` else raise an error

        Output:
        ---------
        count : int
            the count of tailing values in gene  
        """

        ## to find value counts in tail
        ## reverse it and find values in the head
        reversed_string_gene = string_gene[::-1]

        count = self.__head(reversed_string_gene, value)

        return count


    def __head(self, string_gene, value):
        """
        find count of values with `value` in gene in the head of it

        Parameters:
        ------------
        string_gene : string
            a string of zeros and ones representing chromosomes of a gene
        value : integer
            value of chromosome
            shoud be either `0` or `1` else raise an error
        
        Output:
        ---------
        count : int
            the count of heading values in gene  
        """
        
        ## check if the value is right
        self.__check_chromosome(value)

        ## to find the count of chromosomes with value `1` we can count the zeros in the head of gene
        not_value = '0' if value == '1' else '1'

        ## index of the value, can be also assumed as count
        count = string_gene.find(not_value)

        ## if not any chromosome with `value` was found!
        if count == -1:
            count = 0

        return count


    def __check_chromosome(self, chromosome):
        """
        check the chromosome be either zero or one

        Parameters:
        ------------
        chromosome : binary 0 or 1 string
        """
        if not(chromosome == '0' or chromosome == '1'):
            print(chromosome, type(chromosome))
            raise ValueError(f"Chromosome in the gene should be either 0 or 1!\n chromosome is now {chromosome}")

        