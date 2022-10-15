"""
 Saving the results of the generation algorithm into a file
"""
import numpy as np
from algorithm_config import SAVING_STATISTICS_INTERVAL, RESULTS_FILE_NAME


## save the generation fitnesses and when SAVING_STATISTICS_INTERVAL comes, flush it all into a file
GENERATION_FITNESS_STATS_ARR = []

def generation_fitness_save(population_fitness, generation_number) -> None:
    """
    Save the generation fitness values into memory and if the generation number comes to the condition of saving the file, then save the results into disk
    """

    ## writing file headers for the first time
    if generation_number == 0:
        write_file_headers()
        
    ## save cache
    cache_fitness_statistics(population_fitness, generation_number)

    ## and also if the time comes, save them all in a file and delete the cache
    if (generation_number % SAVING_STATISTICS_INTERVAL == 0) & generation_number != 0:
        save_fitness_stats_into_disk(GENERATION_FITNESS_STATS_ARR)


def cache_fitness_statistics(population_fitness, generation_number) -> None:
    """
    function that is to save population fitnesses statistics values in memory
    values are as:
        `generation_number`,
        `min_fitness`,
        `max_fitness`,
        `mean_fitness`,
        `median_fitness`,
        `quarter_fitness`,
        `three_quarter_fitness`
        `standard_deviation`
    """

    min_fitness = np.min(population_fitness)
    max_fitness = np.max(population_fitness)
    mean_fitness = np.mean(population_fitness)
    median_fitness = np.median(population_fitness)
    quarter_fitness = np.percentile(population_fitness, 25)
    three_quarter_fitness = np.percentile(population_fitness, 75)
    standard_deviation = np.std(population_fitness)


    GENERATION_FITNESS_STATS_ARR.append([ generation_number,
                                         min_fitness,
                                         max_fitness,
                                         mean_fitness,
                                         median_fitness,
                                         quarter_fitness,
                                         three_quarter_fitness,
                                         standard_deviation])

def save_fitness_stats_into_disk(generation_fitness_arr,file_name=RESULTS_FILE_NAME) -> None:
    """
    save the fitness stats into disk
    """
    ## convert the array into string with seperation of `,` and save the result into a file
    string_data = ('\n'.join(map(str, generation_fitness_arr)) + '\n').replace(']', '').replace('[', '') 

    with open(file_name, mode='a') as file:
        file.write(string_data)
    
    ## empty the array
    generation_fitness_arr = []

def write_file_headers(file_name=RESULTS_FILE_NAME) -> None:
    """
    write the file headers
    """
    file_headers = ['generation_number', 'min_fitness', 'max_fitness', 'mean_fitness'
                    'median_fitness', 'quarter_fitness', 'three_quarter_fitness', 'standard_deviation']
    file_headers_string = str(file_headers).replace(']', '').replace('[', '') 

    with open(RESULTS_FILE_NAME, mode='w') as file:
        file.write(file_headers_string)