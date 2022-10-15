###################################################
################# Hyperparameters #################
###################################################


from operations.selection import binary_tournament as SELECTION_METHOD
from operations.recombination.binary_recombination import uniform as RECOMBINATION_METHOD
from operations.mutation.binary_mutation import bit_flipping as MUTATION_METHOD

from fitness.binary_fitness import fourpeaks as FITNESS_FUNCTION

from algorithm_config import (PROBLEM_SIZE, 
                                POP_SIZE,
                                P_C,
                                P_M,
                                MAX_GENERATION_COUNT,
                                T)
from algorithm_config import BEST_FITNESS_FOUR_PEAKS as BEST_FITNESS_VALUE
from utils.file_saving import generation_fitness_save
import numpy as np


