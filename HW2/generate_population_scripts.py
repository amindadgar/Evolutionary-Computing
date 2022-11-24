import random
import numpy as np

EVALUATION_COUNT = 0

def generate_vehicles_chromosomes(dataset, MAX_CAPACITY, depot_location_dict ,MAX_DISTANCE, vehicle_depot_constraint):
    """
    Generate chromosomes of vehicles from customers count (data points are summed with 100 in order to be able to know whether the genes in chromosome )

    """
    ## if the depots were more than one
    ## choose randomly from one of it
    ## if the vehicle_depot_constraint was True, don't try to generate more in the loop
    depot_symbol = random.choice(list(depot_location_dict.keys()))
    depot_location = depot_location_dict[depot_symbol]

    

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
    
    max_customer_number = max(dataset.number.values)
    ## not served customers array
    arr = np.linspace(1, max_customer_number, max_customer_number, dtype=int)
    arr = list(arr)

    gene = depot_symbol
    capacity = 0
    distance = 0    

    chromsome_generation_loop_count = 0
    while len(arr) > 0:
        chromsome_generation_loop_count += 1

        ## if it couldn't make the chromsome by serving all cusomers, then break the loop
        if chromsome_generation_loop_count > max_customer_number * 20:
            break

        customer_number = sampling_function(arr, without_replacement=False)
        data = dataset[dataset.number == customer_number]
        capacity += data.demand.values[0]
        
        distance_to_depo = distance + abs(last_X - depot_location[0]) + abs(last_Y - depot_location[1])

        distance += abs(data.x.values[0] - last_X) + abs(data.y.values[0] - last_Y)
        
        
        ## if we couldn't serve the customer we reached the max_distance
        ## also we should check the distance to depot, 
        ## if going back to depot does not reach the limit, then go back to depot
        if distance >= max_distance and distance_to_depo <= max_distance:
            gene += depot_symbol

            ## reset the distance that the vehicle gone
            distance = 0

            last_X, last_Y = depot_location

        ## if going back to depot become over the limit, remove the last customer served from schedule and instead of that customer go back to depot  
        elif distance >= max_distance and distance_to_depo > max_distance:
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

        ## if everything were normal and we could afford the limitations 
        else:
            last_X, last_Y = data.x.values[0], data.y.values[0]
            gene += str(data.number.values[0] + 100) 
            ## if the customer was added to the chromsome, then remove it from sampling array
            arr.remove(customer_number)
        
        ## if there was not constraint on getting back to the same depot
        ## then simply generate another
        if not vehicle_depot_constraint:
            depot_symbol = random.choice(list(depot_location_dict.keys()))
            depot_location = depot_location_dict[depot_symbol]
    
    gene += depot_symbol if gene[-3:] != depot_symbol else ''
    
    return gene

def divide_vehicles(chromosome, vehicle_count, depot_location_dict, max_distance_constraint, vehicle_depot_constraint):
    """
    add the `|` as a division for a chromsome to make different vehicles
    `max_distance_constraint` is a boolean variable which says we have the max_distance constraint or not
    """
    ## initialize the variable
    vehicle_chromsome = None
    all_depots_symbol = [depot_symbol for depot_symbol in depot_location_dict.keys()]

    ## if max distance constraint was not available
    if not max_distance_constraint:
        ## division point is the depots
        ## one less vehicle count should be the division points
        
        each_depot_counts = [chromosome.count(depot_symbol) for depot_symbol in all_depots_symbol]
        depot_count = sum(each_depot_counts)
        sample_arr = np.linspace(1, depot_count - 2, depot_count - 1, dtype=int )
        sample_arr = list(sample_arr)
        division_point = []

        condition_iteration = vehicle_count
        if vehicle_count > len(sample_arr):
            condition_iteration = len(sample_arr)
            
        # for _ in range(vehicle_count - 1):
        for _ in range(condition_iteration):
            point = None
            try:
                point = sampling_function(sample_arr)
            except ValueError as e:
                print(f"Exception: {e}, \nvehicle count: {vehicle_count}, chromosome.count(depot_symbol): {depot_count}")
                quit()
            division_point.append(point)
            
        ## making the copy of string
        vehicle_chromsome = chromosome

        for point in division_point:
            ## the process of finding point-th depot in chromsome
            occurance = -1
            occurance_idx = 0
            while occurance < point:
                ## find each depot occurance and get the minimum of it
                occurances_arr = []
                for depot_symbol in all_depots_symbol:
                    occurance_idx = vehicle_chromsome.find(depot_symbol, occurance_idx) + 1
                    occurances_arr.append(occurance_idx)
                # occurance_idx = vehicle_chromsome.find(depot_symbol, occurance_idx) + 1
                occurance_idx = min(occurances_arr)
                # print(occurances)
                occurance += 1
            
            vehicle_chromsome = vehicle_chromsome[:occurance_idx+2] + '|' + vehicle_chromsome[occurance_idx+2:]
    ## if the constraint was the distance
    ## just partition the chromosome using the depots symbol  
    else:
        ## if the vehicle was limited to one depot
        if vehicle_depot_constraint:

            ## to find the location and the symbol of the depot
            depot_symbol = find_depot_using(chromosome, list(depot_location_dict.keys()))

            ## the index for vehicle
            ## to check whether we reached the limit of vehicles or not
            vehicle_no = 0
            ## start was from depot
            vehicle_chromsome = depot_symbol
            splitted_chromsomes_arr = chromosome.split(depot_symbol)
            ## for splitted chromosome
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
        ## else, there wasn't a constraint on using one depot
        else:
            ## split the array based on different depot symbols
            splitted_chromsomes_arr = []
            vehicle_chromsome = ''
            depots_arr = []
            for idx in range(3, len(chromosome), 3):
                if chromosome[idx-3:idx] not in all_depots_symbol:
                    vehicle_chromsome += chromosome[idx-3:idx]
                else:
                    splitted_chromsomes_arr.append(vehicle_chromsome)
                    ## reset the variable
                    vehicle_chromsome = ''
                    ## save what the depots order was
                    depots_arr.append(chromosome[idx-3:idx])

            ## for splitted chromosome
            for idx, splitted_chromosome in enumerate(splitted_chromsomes_arr):
                ## if it wasn't empty string
                if splitted_chromosome:
                    vehicle_chromsome += splitted_chromosome + depots_arr[idx]
                    vehicle_no += 1
                    ## if we did not reached the limit of vehicles
                    if vehicle_no != vehicle_count:
                        vehicle_chromsome += '|'
                    ## if we reached the number of vehicles in the chromosome
                    else:
                        break

    return vehicle_chromsome

def sampling_function(array, without_replacement=True):
    """
    get some value with or without replacement from array
    """
    sampled_value = random.sample(array, 1)
    if without_replacement:
        array.remove(sampled_value[0])

    return sampled_value[0]

def find_depot_using(chromosome, depot_symbols_arr):
    """
    find the depot that the vehicle is using 
    
    """
    ## to check the chromsome is whether using one or more depots
    depot_count = 0

    ## find the depot, related to the chromosome
    depot_symbol = None
    for depot in depot_symbols_arr:
        depot_symbol_availability = chromosome.find(depot)
        ## if it was available, then put the depot symbol
        if depot_symbol_availability != -1:
            depot_count += 1
            # if depot_count > 1:
            #     raise ValueError(f"Chromsome is using more than one depot!, the last found depot is: {depot}")
            depot_symbol = depot
    
    return depot_symbol

def process_division_points(vehicle_chromosome, depot_symbol_arr, vehicle_depot_constraint):
    """
    Add the depot symbol after the division points
    """    
    if vehicle_depot_constraint:
        processed_vehicle_chromsome = vehicle_chromosome
        depot_symbol = find_depot_using(processed_vehicle_chromsome, depot_symbol_arr)

        division_counts = vehicle_chromosome.count('|')

        # for idx in range(division_counts):
        #     occurance = -1
        #     occurance_idx = 0
        #     while occurance < idx:
        #         occurance_idx = processed_vehicle_chromsome.find('|', occurance_idx) + 1
        #         occurance += 1
            # processed_vehicle_chromsome = processed_vehicle_chromsome[:occurance_idx] + depot_symbol + processed_vehicle_chromsome[occurance_idx:]
        
        occurance = 0
        occurance_idx = 0
        while occurance < division_counts:
            occurance_idx = processed_vehicle_chromsome.find('|', occurance_idx) + 1
            occurance += 1

            processed_vehicle_chromsome = processed_vehicle_chromsome[:occurance_idx] + depot_symbol + processed_vehicle_chromsome[occurance_idx:]
    else:
        processed_vehicle_chromsome = ''
        chromsome_arr = vehicle_chromosome.split('|')
        for idx in range(1, len(chromsome_arr)):
            ## if the array index was not empty
            if chromsome_arr[idx]:
                if chromsome_arr[idx - 1][-3:] in depot_symbol_arr:
                    last_visited_depot = chromsome_arr[idx - 1][-3:]
                    processed_vehicle_chromsome += last_visited_depot + chromsome_arr[idx] + '|'
                else:
                    processed_vehicle_chromsome += chromsome_arr[idx]


    return processed_vehicle_chromsome


def evaluate_distance_fitness(chromsome, DEPOT_LOCATION, dataset, depot_symbol):
    """
    evaluate the distance gone for a chromsome containing different vehicle with the splitter symbol `|`

    """
    global EVALUATION_COUNT
    EVALUATION_COUNT += 1
    # print(chromsome)
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

def evaluate_fitness_customers_count(chromosome, DEPOT_LOCATION, dataset, depot_symbol):
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

    return fitness



def evaluate_fitness_customers_served_demands(chromosome, DEPOT_LOCATION, dataset, depot_symbol):
    """
    evaluate the fitness based on the count of customers' need served
    we want to maximize the customers served demands, instead we minimize it by dividing 1 with the value
    (to make the fewer changes in code, we made it a mimization problem as the distance based fitness)

    two inputs `DEPOT_LOCATION` and are not used, we just added it to make it like the other fitness functions 
    """

    global EVALUATION_COUNT
    EVALUATION_COUNT += 1

    ## get the vehicles array
    customers_arr = chromosome.replace('|', '').replace(depot_symbol, '')
    ## the count of customers served in one path from depot to depot
    demands_served = 0
    ## for each customer
    for idx in range(3, len(customers_arr)+1, 3):
        customer_num = int(customers_arr[idx-3:idx]) - 100
        cusomter = dataset[dataset.number == customer_num]

        customer_demand = cusomter.demand.values[0]
        demands_served += customer_demand

    fitness = 1 / demands_served

    return fitness

def evaluate_fitness_vehicle_count(chromosome, DEPOT_LOCATION, dataset, depot_symbol, customers_count=150, max_distance_limit=200):
    """
    evaluate the fitness based on the vehicles used count
    we want to minimize the vehicle count

    two inputs `DEPOT_LOCATION`, `dataset` and `depot_symbol` are not used, we just added it to make it like the other fitness functions 
    """
    ## save it to normalise after using the other evaluation function for multiple time
    ## then bring the value back to its original
    global EVALUATION_COUNT
    eval_count = EVALUATION_COUNT  + 1

    vehicle_count = chromosome.count('|') + 1

    fitness = vehicle_count

    ## if all customers were not served then the chromsome is not applicable and put a high value representing bad fitness
    chromsome_customer_count = len(chromosome.replace('|', '').replace(depot_symbol, '')) / 3 
    if chromsome_customer_count != customers_count:
        # fitness = 9999999
        ## minimization was our goal, so 1 is divided by (customers_count / chromsome_customer_count)
        fitness = vehicle_count + (customers_count / chromsome_customer_count)
    else:
        ## the distance for each vehicle
        for chromsome_vehicle in chromosome.split('|'):
            distance_gone = evaluate_distance_fitness(chromsome_vehicle, DEPOT_LOCATION, dataset, depot_symbol)

            ## if distance_gone was more than limit, then use the fitness_value as unapplicable (a max value in minimization problem) 
            if distance_gone > max_distance_limit:
                fitness = 9999999

    ## bringing the evaluation count back to original
    EVALUATION_COUNT = eval_count 
    
    return fitness


def generate_population(max_capacity, dataset, fitness_function, depot_location_dict, pop_count = 10, vehicle_count=6, max_distance=None, vehicle_depot_constraint=True):
    """
    generate the population of the problem

    Parameters:
    ------------
    depot_location_dict : dictionary
        the dictionary of depots
    max_capacity : int
        the constraint of the problem, maximum capacity of a vehicle
    max_distance : int
        the constraint of the problem, maximum distance each vehicle can go
        default is `None`
    dataset : dataframe
        the whole information about the problem
    pop_count : int
        the count of the population
        default is 10
    vehicle_count : int
        the count of vehicles to use
        default is 6
    fitness_function : function
        the function for evaluating the chromsomes
    vehicle_depot_constraint : bool
        the constraint for belonging each vehicle to one depot
        default is True, meaning the vehicle does always belong to one depot! (can not go to another depot)

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
        chromsome = generate_vehicles_chromosomes(dataset, max_capacity, depot_location_dict, max_distance, vehicle_depot_constraint)
        
        ## if vehicle count was None, generate numebr of vehicles randomly
        if vehicle_count is None:
            ## the maximum vehicle is set as 60 here
            vehicle_count_value = random.randint(5, 100)
        else:
            vehicle_count_value = vehicle_count
        
        ## to find the location and the symbol of the depot
        depot_symbol = find_depot_using(chromsome, list(depot_location_dict.keys()))
        depot_location = depot_location_dict[depot_symbol]

        ## divide it into different part to make it like different vehicles
        # vehicle_chromsome = divide_vehicles(chromsome, vehicle_count_value, depot_symbol, max_distance_constraint= (max_distance is not None))
        vehicle_chromsome = divide_vehicles(chromsome, vehicle_count_value, depot_location_dict, max_distance_constraint= (max_distance is not None), vehicle_depot_constraint = vehicle_depot_constraint)

        ## process the divisions and make sure that the start points has the depot symbol 
        processed_vehicle_chromsome = process_division_points(vehicle_chromsome, list(depot_location_dict.keys()), vehicle_depot_constraint)
        
        ## add to population array
        population_arr.append(processed_vehicle_chromsome)

        ## fitness evaluations
        chromsome_fitness = fitness_function(processed_vehicle_chromsome, depot_location, dataset, depot_symbol)
        fitness_arr.append(chromsome_fitness)
        
    return population_arr, fitness_arr

def get_evalution_count():
    global EVALUATION_COUNT
    return EVALUATION_COUNT
def set_evaluation_count(value = 0):
    global EVALUATION_COUNT
    EVALUATION_COUNT = value
    return True