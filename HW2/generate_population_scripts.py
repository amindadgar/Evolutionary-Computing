import random
import numpy as np

EVALUATION_COUNT = 0

def generate_vehicles_chromosomes(dataset, MAX_CAPACITY, depot_symbol='(1)'):
    """
    Generate chromosomes of vehicles from 100 data point (data points are summed with 100 in order to be able to know whether the genes in chromosome )

    """

    arr = np.linspace(1, 100, 100, dtype=int)
    arr = list(arr)

    gene = depot_symbol
    capacity = 0

    while len(arr) > 0:

        customer_number = without_replacement_sampling(arr)
        data = dataset[dataset.number == customer_number]
        capacity += data.demand.values[0]

        ## if couldn't carry item, return to depot
        if capacity >= MAX_CAPACITY:
            gene += depot_symbol
            ## reset the capacity
            capacity = 0

            ## bring back the customer to our array
            ## since we haven't used it
            arr.append(data.number.values[0])

        else:
            gene += str(data.number.values[0] + 100) 
    gene += depot_symbol if gene[-3:] != depot_symbol else ''
    
    return gene

def divide_vehicles(chromosome, vehicle_count, depot_symbol):
    """
    add the `|` as a division for a chromsome to make different vehicles
    """

    ## division point is the depots
    ## one less vehicle count should be the division points
    sample_arr = np.linspace(1, chromosome.count(depot_symbol) - 2, chromosome.count(depot_symbol) - 1, dtype=int )
    sample_arr = list(sample_arr)
    division_point = []
    for _ in range(vehicle_count - 1):
        point = without_replacement_sampling(sample_arr)
        division_point.append(point)
        
    ## making the copy of string
    vehicle_chromsome = chromosome

    for point in division_point:
        ## the process of finding point-th depot in chromsome
        occurance = -1
        occurance_idx = 0
        while occurance < point:
            occurance_idx = vehicle_chromsome.find(depot_symbol, occurance_idx) + 1
            occurance += 1
        
        vehicle_chromsome = vehicle_chromsome[:occurance_idx+2] + '|' + vehicle_chromsome[occurance_idx+2:]

    return vehicle_chromsome

def without_replacement_sampling(array):
    """
    get some value without replacement from array
    """
    sampled_value = random.sample(array, 1)
    array.remove(sampled_value[0])

    return sampled_value[0]

def process_division_points(vehicle_chromosome, depot_symbol):
    """
    Add the depot symbol after the division points
    """
    processed_vehicle_chromsome = vehicle_chromosome

    division_counts = vehicle_chromosome.count('|')

    for idx in range(division_counts):
        occurance = -1
        occurance_idx = 0
        while occurance < idx:
            occurance_idx = processed_vehicle_chromsome.find('|', occurance_idx) + 1
            occurance += 1
        
        processed_vehicle_chromsome = processed_vehicle_chromsome[:occurance_idx] + depot_symbol + processed_vehicle_chromsome[occurance_idx:]
    
    return processed_vehicle_chromsome


def evaluate_distance_fitness(chromsome, DEPOT_LOCATION, dataset):
    """
    evaluate the distance gone for a chromsome containing different vehicle with the splitter symbol `|`

    """
    global EVALUATION_COUNT
    EVALUATION_COUNT += 1

    depot_x_loc, depot_y_loc = DEPOT_LOCATION

    distance = 0
    for ch in chromsome.split('(1)'):
        if ch and ch != '|':
            ## start is always from a depot
            last_loc_X, last_loc_Y = DEPOT_LOCATION
            for i in range(3, len(ch)+1, 3):
                ## customer number
                customer_no = int(ch[i-3:i]) - 100

                ## finding the exact customer location from the dataset            
                customer = dataset[dataset.number == customer_no]
                customer_X = customer.x.values[0]
                customer_Y = customer.y.values[0]

                ## manhatan distance
                distance += abs(last_loc_X - customer_X) + abs(last_loc_Y - customer_Y)
                ## update the last locations
                last_loc_X, last_loc_Y = customer_X, customer_Y

            ## end is always the depot
            distance += abs(last_loc_X - depot_x_loc) + abs(last_loc_Y - depot_y_loc)
    
    return distance

def generate_population(max_capacity, DEPOT_LOCATION, dataset, depot_symbol = '(1)', pop_count = 10, vehicle_count=6):
    """
    generate the population of the problem

    Parameters:
    ------------
    depot_symbol : string
        the symbol for the depot
    max_capacity : int
        maximum capacity of a vehicle
    dataset : dataframe
        the whole information about the problem

    Returns:
    ----------
    population_arr : array
        the array of different chromsomes
    fitness_arr : array
        the fitnesses corresponding to each chromsome
    """
    ## chromsomes array
    population_arr = []

    ## fitness array
    fitness_arr = []

    for _ in range(pop_count):
        
        ## just make chromsomes with depot symbols
        chromsome = generate_vehicles_chromosomes(dataset, max_capacity, depot_symbol)
        ## divide it into different part to make it like different vehicles
        vehicle_chromsome = divide_vehicles(chromsome, vehicle_count, depot_symbol)
        ## process the divisions and make sure that the start points has the depot symbol 
        processed_vehicle_chromsome = process_division_points(vehicle_chromsome, depot_symbol)
        
        ## add to population array
        population_arr.append(processed_vehicle_chromsome)

        ## fitness evaluations
        chromsome_fitness = evaluate_distance_fitness(processed_vehicle_chromsome, DEPOT_LOCATION, dataset)
        fitness_arr.append(chromsome_fitness)
        
    return population_arr, fitness_arr

def get_evalution_count():
    global EVALUATION_COUNT
    return EVALUATION_COUNT
def set_evaluation_count(value = 0):
    global EVALUATION_COUNT
    EVALUATION_COUNT = value
    return True