import random
import numpy as np

EVALUATION_COUNT = 0

def generate_vehicles_chromosomes(dataset, MAX_CAPACITY, depot_location ,depot_symbol='(1)', MAX_DISTANCE=None):
    """
    Generate chromosomes of vehicles from 100 data point (data points are summed with 100 in order to be able to know whether the genes in chromosome )

    """
    last_X = depot_location[0]
    last_Y = depot_location[1]

    max_capacity = MAX_CAPACITY
    max_distance = MAX_DISTANCE
    if MAX_CAPACITY is None:
        max_capacity = np.inf
    elif MAX_DISTANCE is None:
        max_distance = np.inf
    else:
        raise ValueError("Both max_distance and max_capacity are None!")
    
    ## not served customers array
    arr = np.linspace(1, 100, 100, dtype=int)
    arr = list(arr)

    gene = depot_symbol
    capacity = 0
    distance = 0    

    while len(arr) > 0:

        customer_number = without_replacement_sampling(arr)
        data = dataset[dataset.number == customer_number]
        capacity += data.demand.values[0]
        
        ## TODO: for multiple depot, check the previous assigned depot
        ## if no depot before, then randomly select one
        distance_to_depo = distance + abs(last_X - depot_location[0]) + abs(last_Y - depot_location[1])

        distance += abs(data.x.values[0] - last_X) + abs(data.y.values[0] - last_Y)
        
        
        ## if we couldn't serve the customer we reached the max_distance
        ## also we should check the distance to depot, 
        ## if going back to depot does not reach the limit, then go back to depot
        # print(distance,'\n', max_distance,'\n',  distance_to_depo)
        if distance >= max_distance and distance_to_depo <= max_distance:
            gene += depot_symbol
            
            ## reset the distance that the vehicle gone
            distance = 0

            ## bring back the customer to our array
            ## since we haven't served him
            arr.append(data.number.values[0])
            last_X, last_Y = depot_location

        ## if going back to depot become over the limit, remove the last customer served from schedule and instead of that customer go back to depot  
        elif distance_to_depo > max_distance:
            ## add the customer to our not served customers list
            arr.append(int(gene[-3:]) - 100)
            ## remove the last customer from gene
            gene = gene[:-3]
            ## replace it with the depot symbol
            gene += depot_symbol
            ## reset the distance that the vehicle gone
            distance = 0

            last_X, last_Y = depot_location
            
        ## if couldn't carry item, return to depot
        elif capacity > max_capacity:
            gene += depot_symbol
            ## reset the capacity or the distance
            capacity = 0

            ## bring back the customer to our array
            ## since we haven't served him
            arr.append(data.number.values[0])
        ## if everything were normal and we could afford the limitations 
        # elif capacity <= max_capacity and distance < max_distance:
        else:
            last_X, last_Y = data.x.values[0], data.y.values[0]
            gene += str(data.number.values[0] + 100) 
    
    gene += depot_symbol if gene[-3:] != depot_symbol else ''
    
    return gene

def divide_vehicles(chromosome, vehicle_count, depot_symbol, max_distance_constraint=False):
    """
    add the `|` as a division for a chromsome to make different vehicles
    `max_distance_constraint` is a boolean variable which says we have the max_distance constraint or not
    """
    ## initialize the variable
    vehicle_chromsome = None
    ## if max distance constraint was not available
    if not max_distance_constraint:
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
    ## if the constraint was the distance
    ## just partition the chromosome with the depots symbol  
    else:
        ## the index for vehicle
        ## to check whether we reached the limit of vehicles or not
        vehicle_no = 0
        ## start was from depot
        vehicle_chromsome = depot_symbol
        splitted_chromsomes_arr = chromosome.split(depot_symbol)
        ## for splitted chromosom
        for splitted_chromosome in splitted_chromsomes_arr:
            ## if it wasn't empty string
            if splitted_chromosome:
                vehicle_chromsome += splitted_chromosome + depot_symbol
                vehicle_no += 1
                ## if we did not reached the limit of vehicles
                if vehicle_no != vehicle_count:
                    vehicle_chromsome += '|'
                ## if we reached the number of vehicles in the chromosome
                else:
                    break

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


def evaluate_distance_fitness(chromsome, DEPOT_LOCATION, dataset, depot_symbol='(1)'):
    """
    evaluate the distance gone for a chromsome containing different vehicle with the splitter symbol `|`

    """
    global EVALUATION_COUNT
    EVALUATION_COUNT += 1

    depot_x_loc, depot_y_loc = DEPOT_LOCATION

    distance = 0
    for ch in chromsome.split(depot_symbol):
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

def evaluate_fitness_customers(chromosome, DEPOT_LOCATION, dataset, depot_symbol='(1)'):
    """
    evaluate the fitness based on the count of customers served
    we want to maximize the customers count, instead we minimize the 1 divided by the customers count
    (to make the fewer changes in code, we made it a mimization problem as the distance based fitness)

    two inputs `DEPOT_LOCATION` and `dataset` are not used, we just added it to make it like the other fitness functions 
    """

    global EVALUATION_COUNT
    EVALUATION_COUNT += 1

    ## get the vehicles array
    vehicles_arr = chromosome.split('|')
    ## the count of customers served in one path from depot to depot
    customers_seved = 0
    ## for each vehicle
    for vehicle in vehicles_arr:
        ## find all the paths for served customers from depot to depot
        path_arr = vehicle.split(depot_symbol)
        ## find the longest path
        longest_path = max(path_arr, key=len)
        ## the count of customers served in the longest path
        customers_seved += len(longest_path) / 3

    fitness = 1 / customers_seved

    ## make the chromsome raw == remove the chromsome depot symbols and division points
    # raw_chromosome = chromosome.replace(depot_symbol, '').replace('|', '')
    ## each customer had 3 letters in chromosome
    # customers_served_count = len(raw_chromosome) / 3

    return fitness


def generate_population(max_capacity, DEPOT_LOCATION, dataset, depot_symbol = '(1)', pop_count = 10, vehicle_count=6, max_distance=None):
    """
    generate the population of the problem

    Parameters:
    ------------
    depot_symbol : string
        the symbol for the depot
    max_capacity : int
        the constraint of the problem, maximum capacity of a vehicle
    max_distance : int
        the constraint of the problem, maximum distance each vehicle can go
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
        chromsome = generate_vehicles_chromosomes(dataset, max_capacity, DEPOT_LOCATION, depot_symbol, max_distance)
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