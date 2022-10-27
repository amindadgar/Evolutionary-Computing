###################################################
################# Hyperparameters #################
###################################################


PROBLEM_SIZE = 10
MAX_GENERATION_COUNT = 300

### specify the json file name here
# JSON_FILE_NAME = "Q3_run_order.json"
JSON_FILE_NAME = "Q3_GreenBoundingBox.json"


## saving the fitness values statistics in result
## how many iteration once to save the results
## default is to set the tenth division of the generation count
SAVING_STATISTICS_INTERVAL = MAX_GENERATION_COUNT // 10


## For fourpeaks or sixpeak fitness functions
## it would not read from here, it would be 0.1 * problem_size
# T = 1

from utils.file_saving import generation_fitness_save
import numpy as np


