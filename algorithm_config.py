PROBLEM_SIZE = 10
POP_SIZE = 200
P_C = 0.7
P_M = 0.5
MAX_GENERATION_COUNT = 10

## saving the fitness values statistics in result
## how many iteration once to save the results
## default is to set the tenth division of the generation count
SAVING_STATISTICS_INTERVAL = MAX_GENERATION_COUNT // 10

## For fourpeaks or sigpeak fitness functions
T = 4

## functions best fitness values
## no need to change this values
## they are set as the max values
BEST_FITNESS_ONE_MAX = PROBLEM_SIZE
BEST_FITNESS_PEAK = 1
BEST_FITNESS_FLIP_FLOP = PROBLEM_SIZE - 1
BEST_FITNESS_FOUR_PEAKS = 2 * PROBLEM_SIZE
BEST_FITNESS_SIX_PEAKS = 2 * PROBLEM_SIZE
BEST_FITNESS_TRAP = 2 * PROBLEM_SIZE

BEST_FITNESS_F1 = 1
BEST_FITNESS_F2 = PROBLEM_SIZE
BEST_FITNESS_F3 = 0

