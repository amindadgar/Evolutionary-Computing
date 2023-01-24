import random
import numpy as np
import re

EVALUATION_COUNT = 0
## the maximum distance in the map that could happen
## initialize the value as None
## it will be set later
MAX_DISTANCE_POSSIBLE = None

def generate_vehicles_chromosomes(dataset, MAX_CAPACITY, depot_location_dict ,MAX_DISTANCE, vehicle_depot_constraint, all_customers=True):
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
    if all_customers:
        random_value = 0
    else:
        random_value = random.randint(10, 80)

    chromsome_generation_loop_count = 0
    while len(arr) - random_value > 0:
        chromsome_generation_loop_count += 1

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
                
        ## if the loop continued for a long time
        if (chromsome_generation_loop_count > max_customer_number * 10):
            ## if it couldn't make the chromsome by serving all cusomers
            ## then just deny the available constraint and add it
            ## is not probable to happen always
            if all_customers:
                if customer_number in arr:
                    last_X, last_Y = data.x.values[0], data.y.values[0]
                    gene += str(data.number.values[0] + 100) 
                    arr.remove(customer_number)
            ## else just break the loop
            else:
                break
        
        
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
    ## probably the capicity constraint is available
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
        ## if the vehicle was limited to use one depot
        # if vehicle_depot_constraint:
        #     ## making the copy of string
        #     vehicle_chromsome = chromosome

        #     for point in division_point:
        #         ## the process of finding point-th depot in chromsome
        #         occurance = -1
        #         occurance_idx = 0
        #         while occurance < point:
        #             ## find each depot occurance and get the minimum of it
        #             occurances_arr = []
        #             for depot_symbol in all_depots_symbol:
        #                 occurance_idx = vehicle_chromsome.find(depot_symbol, occurance_idx) + 1
        #                 occurances_arr.append(occurance_idx)
                    
        #             occurance_idx = min(occurances_arr)
        #             print(occurances_arr)
        #             occurance += 1
                
        #         vehicle_chromsome = vehicle_chromsome[:occurance_idx+2] + '|' + vehicle_chromsome[occurance_idx+2:]
        # else:
        ## a set of points that we can divide the string from
        able_to_divide_arr = []

        ## making the copy of string
        vehicle_chromsome = chromosome

        for depot_symbol in all_depots_symbol: 
            depot_array = [m.start() for m in re.finditer(depot_symbol.replace('(', '\(').replace(')', '\)'), vehicle_chromsome)]
            able_to_divide_arr.extend(depot_array)

        ## remove if the first and last points are in array
        able_to_divide_arr.remove(min(able_to_divide_arr))
        able_to_divide_arr.remove(max(able_to_divide_arr))

        ## the points to divide the vehicle from
        try:
            points_to_divide = random.sample(able_to_divide_arr, vehicle_count)
            points_to_divide = sorted(points_to_divide)
        except ValueError as error:
            ## if less than vehicle count depots available, then just stick to thoes depots available
            points_to_divide = sorted(able_to_divide_arr)
        
        ## adding a symbol to string would change the places of each point
        ## so by adding each symbol we increase the index value for each
        for idx, point in enumerate(points_to_divide):
            vehicle_chromsome = vehicle_chromsome[:point+idx+3] + '|' + vehicle_chromsome[point+idx+3:]                

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
            ## sort based on the length (length divided by 3 can be assumed the customer count here!)
            splitted_chromsomes_arr = sorted(splitted_chromsomes_arr, key=len, reverse=True)
            ## for splitted chromosome
            for splitted_chromosome in splitted_chromsomes_arr:
                ## if it wasn't empty string
                if splitted_chromosome:
                    vehicle_chromsome += splitted_chromosome + depot_symbol
                    vehicle_no += 1
                    ## if we did not reached the limit of vehicles
                    ## if vehicle count was None, then we don't have any limit, so use another vehicle to serve customers
                    if (vehicle_no != vehicle_count) or (vehicle_count is None):
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


def evaluate_distance_fitness(chromsome, DEPOT_LOCATION, dataset, normalize=True):
    """
    evaluate the distance gone for a chromsome containing different vehicle with the splitter symbol `|`
    the returned value is normalized to 1 
    """
    global EVALUATION_COUNT
    EVALUATION_COUNT += 1

    global MAX_DISTANCE_POSSIBLE
    if MAX_DISTANCE_POSSIBLE is None and normalize:
        MAX_DISTANCE_POSSIBLE = find_max_distance_available(dataset, DEPOT_LOCATION)
    
    depot_x_loc, depot_y_loc = DEPOT_LOCATION

    distance = 0 
    
    for ch in re.split('\(\d+\)', chromsome):
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
    
    if normalize:
        ## normalizing
        if distance > MAX_DISTANCE_POSSIBLE:
            raise ValueError("distance cannot be bigger than max distance possible!")
        distance = distance / MAX_DISTANCE_POSSIBLE
        distance *= 100

    
    return distance

def find_max_distance_available(dataset, depot_loc ,distance_metric='manhatan'):
    """
    find the maximum distance that is possible to happen
    not precise but can be an approximation of maximum distance
    randomly a depot location will be used
    """
    max_distance = 0
    
    if distance_metric == 'manhatan':
        for customer in dataset.iterrows():
            max_distance += np.sum(np.abs(customer[1][['x', 'y']] - depot_loc))
    else:
        raise NotImplementedError
    
    return max_distance * 10

def evaluate_fitness_customers_count(chromosome, DEPOT_LOCATION, dataset, normalize=True):
    """
    evaluate the fitness based on the count of customers served
    we want to maximize the customers count, instead we minimize the 1 divided by the customers count but if normalize wasn't requested then do not divide
    (to make the fewer changes in code, we made it a mimization problem as the distance based fitness)

    two inputs `DEPOT_LOCATION` and `dataset` are not used, we just added it to make it like the other fitness functions 
    the returned value is normalized to 1
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
        path_arr = re.split('\(\d+\)', vehicle)
        
        ## find the longest path
        # longest_path = max(path_arr, key=len)
        for path in path_arr:
            ## the count of customers served in the longest path
            customers_seved += len(path) / 3

    ## if original customers was not requested
    if normalize:
        fitness = 1 / customers_seved
        ## to resolve the floating point issues
        # fitness = fitness * 100
    else:
        fitness = customers_seved

    return fitness



def evaluate_fitness_customers_served_demands(chromosome, DEPOT_LOCATION, dataset, normalize=True):
    """
    evaluate the fitness based on the count of customers' need served
    we want to maximize the customers served demands, instead we minimize it by dividing 1 with the value but if normalize was requested then do not divide
    (to make the fewer changes in code, we made it a mimization problem as the distance based fitness)

    two inputs `DEPOT_LOCATION` and are not used, we just added it to make it like the other fitness functions 
    the returned value is normalized to 1
    """

    global EVALUATION_COUNT
    EVALUATION_COUNT += 1

    ## get the vehicles array
    customers_arr = chromosome.replace('|', '')
    ## replace the depot symbols with empty string
    customers_arr = re.sub('\(\d+\)', '', customers_arr)
    ## the count of customers served in one path from depot to depot
    demands_served = 0
    ## for each customer
    for idx in range(3, len(customers_arr)+1, 3):
        customer_num = int(customers_arr[idx-3:idx]) - 100
        cusomter = dataset[dataset.number == customer_num]

        customer_demand = cusomter.demand.values[0]
        demands_served += customer_demand

    if normalize:
        fitness = 1 / demands_served
        ## to resolve the floating point issues
        # fitness = fitness * 100
    else:
        fitness = demands_served

    return fitness

def evaluate_fitness_vehicle_count(chromosome, DEPOT_LOCATION, dataset, customers_count=150, max_distance_limit=200):
    """
    evaluate the fitness based on the vehicles used count
    we want to minimize the vehicle count

    the returned value is normalized to 1
    """
    ## save it to normalise after using the other evaluation function for multiple time
    ## then bring the value back to its original
    global EVALUATION_COUNT
    eval_count = EVALUATION_COUNT  + 1

    vehicle_count = chromosome.count('|') + 1

    distance_over = find_distance_penalty(chromosome, DEPOT_LOCATION, dataset, max_distance_limit, vehicle_count)

    ## find the mean of the distances over the limit
    mean_distances_over = distance_over / vehicle_count

    ## to make it as a float value
    mean_distances_over = mean_distances_over / max_distance_limit

    ## we can after find out how many kilometers more than limit has been gone by the floating value  
    ## A penalty value 
    fitness = vehicle_count + mean_distances_over

    ## bringing the evaluation count back to original
    EVALUATION_COUNT = eval_count 
    
    return fitness

def find_distance_penalty(chromosome, DEPOT_LOCATION, dataset, max_distance_limit, vehicle_count):
    """
    find the distance gone over the maximum distance, and return a floating value representing the penalty of distance gone over the max distance
    """

    ## to measure if more distances has been gone 
    distance_over = 0
    ## the distance for each vehicle
    for chromsome_vehicle in chromosome.split('|'):
        distance_gone = evaluate_distance_fitness(chromsome_vehicle, DEPOT_LOCATION, dataset, normalize=False)
        
        ## sum the more distances over the limit
        if distance_gone > max_distance_limit:
            distance_over += distance_gone - max_distance_limit
    
    return distance_over

def fitness_sharing(fitness):
    """
    update the fitness values using fitness sharing method
    this is a variety preservation method for EC algorithms
    """
    raise NotImplementedError



def generate_population(max_capacity, 
                        dataset, 
                        fitness_functions, 
                        depot_location_dict, 
                        pop_count = 10, 
                        vehicle_count=6, 
                        max_distance=None, 
                        vehicle_depot_constraint=True, 
                        all_customers=True,
                        multi_objective_handler = 'coeff'):
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
    fitness_functions : function
        the function for evaluating the chromsomes
        if a list, then the problem is multi-objective, else it is single objective
        can be None, meaning there is no need for fitness values
    vehicle_depot_constraint : bool
        the constraint for belonging each vehicle to one depot
        default is True, meaning the vehicle does always belong to one depot! (can not go to another depot)
    multi_objective_handler : string
        the handler for showing which type of method is used for multi-objective algorithm
        can be 'coeff' or 'pareto', default is 'coeff'

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
        chromsome = generate_vehicles_chromosomes(dataset, max_capacity, depot_location_dict, max_distance, vehicle_depot_constraint, all_customers)
        
        ## to find the location and the symbol of the depot
        depot_symbol = find_depot_using(chromsome, list(depot_location_dict.keys()))
        depot_location = depot_location_dict[depot_symbol]

        ## divide it into different part to make it like different vehicles
        # vehicle_chromsome = divide_vehicles(chromsome, vehicle_count_value, depot_symbol, max_distance_constraint= (max_distance is not None))
        vehicle_chromsome = divide_vehicles(chromsome, vehicle_count, depot_location_dict, max_distance_constraint= (max_distance is not None), vehicle_depot_constraint = vehicle_depot_constraint)

        ## process the divisions and make sure that the start points has the depot symbol 
        processed_vehicle_chromsome = process_division_points(vehicle_chromsome, list(depot_location_dict.keys()), vehicle_depot_constraint)
        
        ## add to population array
        population_arr.append(processed_vehicle_chromsome)

        ## fitness evaluations
        ## if there was a single fitness function
        if type(fitness_functions) is not list and fitness_functions is not None:
            chromsome_fitness = fitness_functions(processed_vehicle_chromsome, depot_location, dataset)
        ## else we had multiple fitness functions
        elif fitness_functions is not None:
            if multi_objective_handler == 'coeff':
                chromsome_fitness = multi_objective_fitness_coeff(fitness_functions, processed_vehicle_chromsome, depot_location, dataset)
            else:
                chromsome_fitness = multi_objective_fitness(fitness_functions, processed_vehicle_chromsome, depot_location, dataset)
        else:
            ## if no fitness function was given
            ## meaning no evaluation needed 
            chromsome_fitness = None
        fitness_arr.append(chromsome_fitness)
        
    return population_arr, fitness_arr

def multi_objective_fitness_coeff(fitness_functions_arr, chromosome, depot_location, dataset):
    """
    combine objective functions with just coefficients of 1 (positive and negative)
    returns the fitness value obtained from multiple fitness functions
    """

    ## save it to normalise after using the other evaluation function for multiple time
    ## then bring the value back to its original
    global EVALUATION_COUNT
    eval_count = EVALUATION_COUNT  + 1

    ## initialization
    fitness_value = 0

    for idx, fitness_function in enumerate(fitness_functions_arr):
        ## coefficients one in between will be negative
        coeff = (-1) ** idx
        fitness_value += coeff * fitness_function(chromosome, depot_location, dataset)

    EVALUATION_COUNT = eval_count
    
    return fitness_value

def multi_objective_fitness(fitness_functions_arr, chromosome, depot_location, dataset):
    """
    Multi-objective fitness values 
    the returned fitness is an array of fitness values representing each fitness function
    """
    ## save it to normalise after using the other evaluation function for multiple time
    ## then bring the value back to its original
    global EVALUATION_COUNT
    eval_count = EVALUATION_COUNT  + 1

    fitness_arr = []
    for fitness_function in fitness_functions_arr:
        fitness = fitness_function(chromosome, depot_location, dataset)
        fitness_arr.append(fitness)

    EVALUATION_COUNT = eval_count

    return fitness_arr    


def get_evalution_count():
    global EVALUATION_COUNT
    return EVALUATION_COUNT
def set_evaluation_count(value = 0):
    global EVALUATION_COUNT
    EVALUATION_COUNT = value
    return True