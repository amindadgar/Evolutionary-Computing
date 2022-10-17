###################################################
################# Hyperparameters #################
###################################################


from operations.selection import binary_tournament as SELECTION_METHOD
from operations.recombination.binary_recombination import single_point as RECOMBINATION_METHOD
from operations.mutation.binary_mutation import bit_flipping as MUTATION_METHOD

  

from fitness.binary_fitness import flipflop as FITNESS_FUNCTION

from algorithm_config import (PROBLEM_SIZE, 
                                POP_SIZE,
                                P_C,
                                P_M,
                                MAX_GENERATION_COUNT,
                                T, RESULTS_FILE_NAME)
from algorithm_config import BEST_FITNESS_ONE_MAX as BEST_FITNESS_VALUE
from utils.file_saving import generation_fitness_save
import numpy as np


