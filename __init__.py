###################################################
################# Hyperparameters #################
###################################################


PROBLEM_SIZE = 10
MAX_GENERATION_COUNT = 300


## saving the fitness values statistics in result
## how many iteration once to save the results
## default is to set the tenth division of the generation count
SAVING_STATISTICS_INTERVAL = MAX_GENERATION_COUNT // 10


## For fourpeaks or sixpeak fitness functions
T = 4

from utils.file_saving import generation_fitness_save
import numpy as np


