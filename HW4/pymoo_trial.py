import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import json
from generate_population_scripts import (generate_population,
                                         evaluate_distance_fitness, 
                                         evaluate_fitness_customers_count,
                                         evaluate_fitness_customers_served_demands,
                                         evaluate_fitness_vehicle_count,
                                         find_depot_using,
                                         get_evalution_count, 
                                         set_evaluation_count,
                                         multi_objective_fitness_coeff, 
                                         multi_objective_fitness)
from combination import cut_and_crossfill, mutation_inverse, mutation_scramble
from selection import roulette_wheel, binary_tournament
from multi_objective_handelers import find_pareto_set
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.crossover import Crossover
from pymoo.core.sampling import Sampling
from pymoo.core.problem import ElementwiseProblem
from generate_population_scripts import find_depot_using
from pymoo.algorithms.moo.nsga2 import NSGA2
import sys


class MyProblem(ElementwiseProblem):

    def __init__(self, max_capacity, 
                        dataset, 
                        fitness_functions, 
                        depot_location_dict, 
                        pop_count = 10, 
                        vehicle_count=6, 
                        max_distance=None, 
                        vehicle_depot_constraint=True, 
                        all_customers=True):

        super().__init__(n_var=1,
                         n_obj=2,
                         n_ieq_constr=0)
          
        self.max_capacity = max_capacity
        self.dataset = dataset
        self.fitness_functions = fitness_functions
        self.depot_location_dict = depot_location_dict
        self.pop_count = pop_count
        self.vehicle_count = vehicle_count
        self.max_distance = max_distance
        self.vehicle_depot_constraint = vehicle_depot_constraint
        self.all_customers = all_customers

    def _evaluate(self, x, out, *args, **kwargs):

        depot_symbol = find_depot_using(x[0], ['(1)', '(2)', '(3)'])
        f1 = evaluate_distance_fitness(x[0], self.depot_location_dict[depot_symbol], self.dataset)
        f2 = evaluate_fitness_customers_count(x[0], self.depot_location_dict[depot_symbol], self.dataset)

        out["F"] = [f1, f2]



class problem1_sampling(Sampling):

    def _do(self, problem, n_samples,**kwargs):

        X, _ = generate_population(
            max_capacity=problem.max_capacity,
            dataset=problem.dataset,
            fitness_functions = None, 
            depot_location_dict = problem.depot_location_dict,
            pop_count=n_samples, 
            vehicle_count=problem.vehicle_count,
            max_distance=problem.max_distance,
            all_customers=problem.all_customers, 
            multi_objective_handler='nsga'
        )

        X = np.array(X).reshape(n_samples, 1)

        return X

class problem1_crossover(Crossover):
    def __init__(self):

        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):
        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape

        # The output owith the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        Y = np.full_like(X, None, dtype=object)

        for k in range(n_matings):

            Y[0, k, 0], Y[1, k, 0] = cut_and_crossfill(X[0, k, 0], X[1, k, 0], problem.dataset, 
                                max_capacity=problem.max_capacity,
                                max_distance=problem.max_distance,
                                depot_location_dict=problem.depot_location_dict,
                                vehicle_depot_constraint=problem.vehicle_depot_constraint,
                                vehicle_count=problem.vehicle_count)

        return Y

class problem1_mutation(Mutation):
    def __init__(self):
        super().__init__()
        self.prob = 0.2

    def _do(self, problem, X, **kwargs):

        n_pop, n_var = X.shape

        Y = np.full_like(X, None, dtype=object)

        for i in range(n_pop):
            Y[i, 0] = mutation_scramble(chromosome= X[i, 0], 
                max_capacity= problem.max_capacity, 
                dataset= problem.dataset, 
                depot_location_dict= problem.depot_location_dict,
                max_distance=problem.max_distance,
                vehicle_depot_constraint= problem.vehicle_depot_constraint,
                vehicle_count= problem.vehicle_count
            )

        return Y

class problem1_duplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        return a.X[0] == b.X[0]



if __name__ == '__main__':

    if len(sys.argv) > 3 or len(sys.argv) < 3:
        raise ValueError(f"One arguments expected, got {len(sys.argv) - 1}")
    
    
    dataset_location = sys.argv[1]
    ## example for the first argument
    ## dataset_loc=data/P1-2.txt
    dataset_location = dataset_location.split('=')[1]

    ## example for the second argument
    ## problem_no=1
    problem_no = int(sys.argv[2].split('=')[1])

    p1_data = pd.read_csv(dataset_location, delimiter=' ')

    if problem_no == 1 or problem_no == 2:
        DEPOT_LOCATIONS_dict = {
                '(1)': (31, 6),
                '(2)': (-31, 7),
                '(3)': (25, -10)
        }   
    elif problem_no == 3:
                DEPOT_LOCATIONS_dict = {
                '(1)': (40, 23),
                '(2)': (0, 0),
                '(3)': (-18, -27)
        }  
    else:
        raise ValueError(f"Wrong problem number entered: {problem_no}")

    algorithm = NSGA2(
        pop_size=50,
        sampling=problem1_sampling(),
        crossover=problem1_crossover(),
        mutation=problem1_mutation(),
        eliminate_duplicates=problem1_duplicateElimination()
    )

    # termination = get_termination("n_gen", 10)
    termination = get_termination("n_eval", 5000)

    problem = MyProblem(max_capacity=100,
                        dataset=p1_data, 
                        fitness_functions=[evaluate_distance_fitness, evaluate_fitness_customers_count],
                        depot_location_dict=DEPOT_LOCATIONS_dict,
                        pop_count=50,
                        vehicle_count=4,
                        max_distance=None,
                        all_customers=False,
                        vehicle_depot_constraint=True)

    res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)

    X = res.X
    F = res.F

    # print('res.X: \n',X)
    print('res.F: \n',F)
