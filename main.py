import sys
from algorithm import execute_program
from operations.selection import binary_tournament, roulette_wheel
from operations.recombination.binary_recombination import uniform, single_point
from operations.mutation.binary_mutation import bit_flipping
from __init__ import T, MAX_GENERATION_COUNT
from fitness.binary_fitness import fitness


## check if we want to run the algorithm mulitple run
arg_counts = len(sys.argv)

algorithm_run_counts = 1
## if the run count was given as arguments
if arg_counts == 2:
    algorithm_run_counts = int(sys.argv[1])

## setting up the fitness functions
fitness_functions = fitness(T)
[onemax, peak, flipflop, fourpeaks, sixpeaks, trap] = [fitness_functions.onemax, 
                                                        fitness_functions.peak, 
                                                        fitness_functions.flipflop, 
                                                        fitness_functions.fourpeaks, 
                                                        fitness_functions.sixpeaks, 
                                                        fitness_functions.trap]
                                                        
execute_program(algorithm_run_counts, 
                FITNESS_FUNCTION={'onemax': onemax,'peak': peak, 'flipflop': flipflop, 'fourpeaks': fourpeaks, 'sixpeaks': sixpeaks, 'trap': trap},
                SELECTION_METHOD= {'roulette_wheel': roulette_wheel}, 
                RECOMBINATION_METHOD={'single_point': single_point}, 
                MUTATION_METHOD={'bit_flipping': bit_flipping}, 
                problem_size_arr=[30], 
                popSizeArr=[100],
                P_m=[0.05, 0.1, 0.3, 0.5],
                P_c=[0.5])