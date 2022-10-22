from __init__ import *
import os


### The best fitness values
BEST_FITNESS_ONE_MAX = PROBLEM_SIZE
BEST_FITNESS_PEAK = 1
BEST_FITNESS_FLIP_FLOP = PROBLEM_SIZE - 1
BEST_FITNESS_FOUR_PEAKS = 2 * PROBLEM_SIZE
BEST_FITNESS_SIX_PEAKS = 2 * PROBLEM_SIZE
BEST_FITNESS_TRAP = 2 * PROBLEM_SIZE

BEST_FITNESS_F1 = 1
BEST_FITNESS_F2 = PROBLEM_SIZE
BEST_FITNESS_F3 = 0

def algorithm_run(file_name, problem_size, best_fitness_value, FITNESS_FUNCTION,SELECTION_METHOD, RECOMBINATION_METHOD, MUTATION_METHOD, pop_size, p_m, p_c ):
    """
    Run the whole algorithm and save the results in file_name

    Parameters:
    -------------
    file_name : string
        the name of the file that the results are saved
    problem_size : int
        the gene length of the algorithm
    best_fitness_value : int
        the best fitness representing the value for the fitness function
    """

    ########### Step 1 & 2 ###########
    ########### Population and their fitness generation ###########

    population = []
    fitness_pop = []

    for _ in range(pop_size):
        gene = str(np.random.randint(0, 2, problem_size)).replace(' ', '').replace('\n', '')[1:problem_size + 1]
        fitness = FITNESS_FUNCTION(gene)
        
        
        population.append(gene)
        fitness_pop.append(fitness)
        # print('Population:\n', population)
    ########### Step 3 ###########
    generation_count = 0
    end_condition = False

    while  not end_condition:
        ## make I/O overhead less
        if generation_count % 10 == 0:
            print(f'Generation number {generation_count}')

        ## Step 4 & 5 
        ## Parents pool is created and randomly they are paired

        parent_pairs = []
        for _ in range(pop_size):
            parent = SELECTION_METHOD(population=population, pop_fitness=fitness_pop, selection_pop_size=2)
            
            parent_pairs.append(parent)

        ## Step 6 & 7 & 8
        ## Apply recombination with P_C probability and mutation with P_M probability
        ## Also find offsprings fitnesses

        offsprings = []
        fitness_offsprings = []
        for parents in parent_pairs:
            recombination_p = np.random.random()

            ## the offspring for this iteration
            iteration_offspring = []
            
            ######## Recombination ########
            if recombination_p < p_c:
                offspring1, offspring2 =  RECOMBINATION_METHOD(parents[0], parents[1])

                iteration_offspring = [offspring1, offspring2]

            
            ######## Mutation ########
            ## if cross over has happend
            if len(iteration_offspring) != 0:
                offspring1 = MUTATION_METHOD(iteration_offspring[0], p_m)
                offspring2 = MUTATION_METHOD(iteration_offspring[1], p_m)

                iteration_offspring = [offspring1, offspring2]
            ## if cross over had not happened
            else:
                offspring1 = MUTATION_METHOD(parents[0], p_m)
                offspring2 = MUTATION_METHOD(parents[1], p_m)

                iteration_offspring = [offspring1, offspring2]


            
            ## finally append the genarated offsprings to offspring array 
            offsprings.append(iteration_offspring[0])
            offsprings.append(iteration_offspring[1])

            fitness_offsprings.append(FITNESS_FUNCTION(iteration_offspring[0]))
            fitness_offsprings.append(FITNESS_FUNCTION(iteration_offspring[1]))

        ## Step 9
        ## replace the old population with the new ones

        ## the whole generation: parents + offsprings
        generation_population = population.copy()
        generation_population.extend(offsprings)

        ## whole generation fitness: parents fitness + offsprings fitness
        generation_fitness = fitness_pop.copy()
        generation_fitness.extend(fitness_offsprings)

        ## the sorted generation
        generation_population_sorted = np.array(generation_population)[np.argsort(generation_fitness)]
        generation_fitness_sorted = np.sort(generation_fitness)

        ## Step 10
        ## extract the best of the new generation

        best_of_generation_population = generation_population_sorted[-200:]
        best_of_generation_fitness = generation_fitness_sorted[-200:]

        ## save them into the original population arrays
        population = best_of_generation_population.tolist()
        fitness_pop = best_of_generation_fitness.tolist()

        # print(population)        
        ## increase the generation value
        generation_count += 1
        

        ## condition checks
        convergence_condition = best_of_generation_fitness[-1] == best_fitness_value
        if best_of_generation_fitness[-1] == 1:
            print(best_of_generation_fitness[-1])
        end_condition = convergence_condition or (generation_count == MAX_GENERATION_COUNT)
        
        ## generation fitness statistics save
        generation_fitness_save(best_of_generation_fitness, generation_count, end_condition, file_name)        

        if convergence_condition:
            print(f"Algorithm converged in {generation_count} generations!")
        elif end_condition:
            print(f"Algorithm did not converged and ended in selected max generation count {generation_count}, best_one={best_of_generation_fitness[-1]}, best_fitness_value={best_fitness_value}!")



def execute_program(algorithm_run_counts, FITNESS_FUNCTION, SELECTION_METHOD, RECOMBINATION_METHOD, MUTATION_METHOD, problem_size_arr, popSizeArr, P_m, P_c ,additional_fileName='') -> None:
    """
    Execute algorithm multiple time

    Parameters:
    ------------
    algorithm_run_counts : int
        specify to run algorithm multiple time 
    FITNESS_FUNCTION : dictionary
        dictionary of values representing each fitness function and the keys are the same as the fitness functions name
    SELECTION_METHOD : dictionary
        dictionary of values representing each fitness function
    """
    
    for mutation_method in MUTATION_METHOD.keys():
        for recombination_method in RECOMBINATION_METHOD.keys():
            for selection_method in SELECTION_METHOD.keys():
                for fitness_function in FITNESS_FUNCTION.keys():
                    
                    for problem_size in problem_size_arr:
                        refresh_fitness_values(problem_size)
                        best_fitness_value = select_best_fitness_value(fitness_function)
                        
                        for probability_cross_over in P_c:
                            for probability_mutation in P_m: 
                                for population_size in popSizeArr:
                                    for i in range(algorithm_run_counts):
                                        
                                        file_name = f'Resutls_fitness-function={fitness_function}_Pc={probability_cross_over}_Pm={probability_mutation}_PopSize={population_size}_ProblemSize={problem_size}_selection-method={selection_method}_recombination-method={recombination_method}_mutation-method={mutation_method}_algorithm_run={i}{additional_fileName}.csv'
                                        file_name = os.path.join('results', file_name)
                                        print(f'{file_name}: ', '\n' + '-'*15)
                                        
                                        algorithm_run(file_name = file_name, 
                                                    problem_size=problem_size,
                                                    best_fitness_value=best_fitness_value, 
                                                    FITNESS_FUNCTION=FITNESS_FUNCTION[fitness_function],
                                                    SELECTION_METHOD=SELECTION_METHOD[selection_method],
                                                    RECOMBINATION_METHOD = RECOMBINATION_METHOD[recombination_method], 
                                                    MUTATION_METHOD = MUTATION_METHOD[mutation_method], 
                                                    pop_size=population_size,
                                                    p_c=probability_cross_over, 
                                                    p_m=probability_mutation)
        


def select_best_fitness_value(fitness_function_name) -> int:
    """
    select the best fitness value corresponding to each fitness function
    """
    if fitness_function_name == 'onemax':
        return BEST_FITNESS_ONE_MAX
    elif fitness_function_name == 'peak':
        return BEST_FITNESS_PEAK
    elif fitness_function_name == 'flipflop':
        return BEST_FITNESS_FLIP_FLOP
    elif fitness_function_name == 'fourpeaks':
        return BEST_FITNESS_FOUR_PEAKS
    elif fitness_function_name == 'sixpeaks':
        return BEST_FITNESS_SIX_PEAKS
    elif fitness_function_name == 'trap':
        return BEST_FITNESS_TRAP
    elif fitness_function_name == 'f1':
        return BEST_FITNESS_F1
    elif fitness_function_name == 'f2':
        return BEST_FITNESS_F2
    elif fitness_function_name == 'f3':
        return BEST_FITNESS_F3
    else:
        raise ValueError(f"Error! incorrect fitness function name: {fitness_function_name}!")
    

def refresh_fitness_values(problem_size):
    """
    Referesh the configs based on the problem size
    """

    global BEST_FITNESS_F1
    global BEST_FITNESS_F2
    global BEST_FITNESS_F3
    global BEST_FITNESS_ONE_MAX
    global BEST_FITNESS_PEAK
    global BEST_FITNESS_FLIP_FLOP
    global BEST_FITNESS_FOUR_PEAKS
    global BEST_FITNESS_SIX_PEAKS
    global BEST_FITNESS_TRAP


    ## functions best fitness values
    ## no need to change this values
    ## they are set as the max values
    BEST_FITNESS_ONE_MAX = problem_size
    BEST_FITNESS_PEAK = 1
    BEST_FITNESS_FLIP_FLOP = problem_size - 1
    BEST_FITNESS_FOUR_PEAKS = 2 * problem_size
    BEST_FITNESS_SIX_PEAKS = 2 * problem_size
    BEST_FITNESS_TRAP = 2 * problem_size

    BEST_FITNESS_F1 = 1
    BEST_FITNESS_F2 = problem_size
    BEST_FITNESS_F3 = 0
