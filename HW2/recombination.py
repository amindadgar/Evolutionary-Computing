import random
from generate_population_scripts import without_replacement_sampling, process_division_points, divide_vehicles


def split_string_step(string, step):
    """
    split a string based on steps
    """
    string_arr = []

    for i in range(1, len(string)+1):
        if i % step == 0:
            string_arr.append(string[i-step:i])
    return string_arr

def cut_and_crossfil(chromosome1, chromosome2, dataset, max_capacity, depot_symbol=['(1)']):
    """
    cut and crossfill cross-over

    at the end repair the chromsome with max_capacity
    """
    ## find the vehicle count
    vehicle_count = chromosome1.count('|') + 1

    ## find the chromsome with minimum length and break it
    chromosome_to_break = chromosome1 if len(chromosome1) < len(chromosome2) else chromosome2
    second_chromsome = chromosome1 if len(chromosome1) > len(chromosome2) else chromosome2

    ## remove all the symbols inside the chromsome
    chromosome = chromosome_to_break.replace('|', '')
    second_chromsome = second_chromsome.replace('|', '')

    for symbol in depot_symbol:
        chromosome = chromosome.replace(symbol, '')
        second_chromsome = second_chromsome.replace(symbol, '')
    
    ## break point of the chromsomes
    ## the break point should not be the first or the last points
    ## create an array and randomly sample from it
    arr = np.arange(3, min(len(chromosome), len(second_chromsome)) - 3, 3)
    break_point = without_replacement_sampling(list(arr))
    break_point = break_point // 3
    
    chromosome_arr = split_string_step(chromosome, 3)
    second_chromsome_arr = split_string_step(second_chromsome, 3)

    ## offspring creation
    offspring1 = chromosome_arr[:break_point]

    for gene in second_chromsome_arr[break_point:] + second_chromsome_arr[:break_point]:
        if gene not in offspring1:
            offspring1.append(gene)
    
    offspring2 = second_chromsome_arr[break_point:]
    for gene in chromosome_arr[:break_point] + chromosome_arr[break_point:] :
        if gene not in offspring2:
            offspring2.append(gene)

    ## repairment
    random_depot = random.sample(depot_symbol, 1)[0]
    offspring1 = process_depots(dataset, offspring1, max_capacity, random_depot)
    offspring1 = divide_vehicles(offspring1, vehicle_count, random_depot)
    offspring1 = process_division_points(offspring1, random_depot)

    random_depot = random.sample(depot_symbol, 1)[0]
    offspring2 = process_depots(dataset, offspring2, max_capacity, random_depot)
    offspring2 = divide_vehicles(offspring2, vehicle_count, random_depot)
    offspring2 = process_division_points(offspring2, random_depot)



    return offspring1, offspring2

def process_depots(dataset ,string_chromsome_arr, max_capacity, depot_symbol):
    """
    Add the depot symbols to string chromsome array (the customer numbers are strings in the array)
    """
    capacity = 0
    idx = 0
    string_chromsome = depot_symbol
    while idx < len(string_chromsome_arr):
        gene = string_chromsome_arr[idx]

        customer_no = int(gene) - 100
        customer = dataset[dataset.number == customer_no]

        capacity += customer.demand.values[0]
        
        if capacity > max_capacity:
            string_chromsome += depot_symbol
            capacity = 0
        else:
            string_chromsome += gene
            idx += 1

    string_chromsome += depot_symbol if string_chromsome[-3:] != depot_symbol else ''
    
    
    return string_chromsome

def mutation(chromsome):
    ## TODO
    pass