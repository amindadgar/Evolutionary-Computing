from __init__ import *


#########################################
################# Start #################
#########################################


def algorithm_run(file_name):
    """
    Run the whole algorithm and save the results in file_name
    """

    ########### Step 1 & 2 ###########
    ########### Population and their fitness generation ###########

    population = []
    fitness_pop = []

    for _ in range(POP_SIZE):
        gene = str(np.random.randint(0, 2, PROBLEM_SIZE)).replace(' ', '')[1:PROBLEM_SIZE + 1]
        fitness = FITNESS_FUNCTION(gene)
        
        
        population.append(gene)
        fitness_pop.append(fitness)
        # print('Population:\n', population)
    ########### Step 3 ###########
    generation_count = 0
    end_condition = False

    # print('-'*5)
    while  not end_condition:
        print(f'Generation number {generation_count}')

        ## Step 4 & 5 
        ## Parents pool is created and randomly they are paired

        parent_pairs = []
        for _ in range(POP_SIZE):
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
            if recombination_p < P_C:
                offspring1, offspring2 =  RECOMBINATION_METHOD(parents[0], parents[1])

                iteration_offspring = [offspring1, offspring2]

            
            ######## Mutation ########
            ## if cross over has happend
            if len(iteration_offspring) != 0:
                offspring1 = MUTATION_METHOD(iteration_offspring[0], P_M)
                offspring2 = MUTATION_METHOD(iteration_offspring[1], P_M)

                iteration_offspring = [offspring1, offspring2]
            ## if cross over was not happened
            else:
                offspring1 = MUTATION_METHOD(parents[0], P_M)
                offspring2 = MUTATION_METHOD(parents[1], P_M)

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
        convergence_condition = str.count(max(population), '1') == BEST_FITNESS_VALUE
        end_condition = convergence_condition or (generation_count == MAX_GENERATION_COUNT)
        
        ## generation fitness statistics save
        generation_fitness_save(best_of_generation_fitness, generation_count, end_condition, file_name)

        if convergence_condition:
            print(f"Algorithm converged in {generation_count} generations!")
            # print(str(max(population)))
        elif end_condition:
            print(f"Algorithm did not converged and ended in selected max generation count {generation_count}!")
