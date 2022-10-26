import sys
from algorithm import execute_program
from __init__ import T
from read_json_config import (run_order_dict,
                                 create_fitness_function_dict,
                                 create_selection_method_dict, 
                                 create_recombination_method_dict, 
                                 create_mutation_method_dict)



## check if we want to run the algorithm mulitple run
arg_counts = len(sys.argv)

algorithm_run_counts = 1
## if the run count was given as arguments
if arg_counts == 2:
    algorithm_run_counts = int(sys.argv[1])
 
for run_id in run_order_dict:
    print('\n', '#' * 15, f' RUN: {run_id} ', '#' * 15 , '\n')

    ## to run one of the orders uncomment below
    # if run_id != 'order_12':
    #     print(f'{run_id} breaked!')
    #     continue

    fitness_function_arr = run_order_dict[run_id]['FITNESS_FUNCTIONS']
    ## uncomment the line below while using binary genes
    FITNESS_FUNCTION_dict = create_fitness_function_dict(fitness_function_arr, T=T)

    ## for integer genes, the original gene needed to be the size of problem_size and given the question in the exercise its size is 10
    ## so we need to multiply it to cover the lengths with more than 10 (since we had max size 15, multiplication with 5 is much more than enough)
    # FITNESS_FUNCTION_dict = create_fitness_function_dict(fitness_function_arr, gene_original=run_order_dict[run_id]['GENE_ORIGNIAL'] * 5)


    SELECTION_METHOD_arr = run_order_dict[run_id]['SELECTION_METHODS']
    SELECTION_METHOD_dict = create_selection_method_dict(SELECTION_METHOD_arr)

    RECMOBINATION_arr = run_order_dict[run_id]['RECOMBINATION_METHODS']
    RECMOBINATION_dict = create_recombination_method_dict(RECMOBINATION_arr)

    MUTATION_arr = run_order_dict[run_id]['MUTATION_METHODS']
    MUTATION_dict = create_mutation_method_dict(MUTATION_arr)

    PROBLEM_SIZES = run_order_dict[run_id]['PROBLEM_SIZES']
    POPULATION_SIZES = run_order_dict[run_id]['POPULATION_SIZES']

    PROBABILITIES_MUTATION = run_order_dict[run_id]['PROBABILITIES_MUTATION']
    PROBABILITIES_CROSSOVER = run_order_dict[run_id]['PROBABILITIES_CROSSOVER']

    execute_program(algorithm_run_counts, 
                    FITNESS_FUNCTION= FITNESS_FUNCTION_dict,
                    SELECTION_METHOD= SELECTION_METHOD_dict, 
                    RECOMBINATION_METHOD= RECMOBINATION_dict, 
                    MUTATION_METHOD= MUTATION_dict, 
                    problem_size_arr= PROBLEM_SIZES, 
                    popSizeArr= POPULATION_SIZES,
                    P_m= PROBABILITIES_MUTATION,
                    P_c= PROBABILITIES_CROSSOVER)   
