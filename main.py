import sys
from __init__ import RESULTS_FILE_NAME
from algorithm import algorithm_run
from algorithm_config import PROBLEM_SIZE


## check if we want to run the algorithm mulitple run
arg_counts = len(sys.argv)

algorithm_run_counts = 1
## if the run count was given as arguments
if arg_counts == 2:
    algorithm_run_counts = int(sys.argv[1])


for size in [10, 30, 50, 100]:
    PROBLEM_SIZE = size
    print('/'*15 + str(PROBLEM_SIZE) + '\\'*15)

    for i in range(algorithm_run_counts):
        print('Algorithm Run: ')
        print('-'*15)
        algorithm_run(RESULTS_FILE_NAME + f'_algorithm_run={i}.csv')