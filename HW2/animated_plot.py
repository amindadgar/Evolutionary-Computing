import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import re
import sys
import pandas as pd
import json
import os

DATA_TO_PLOT = []
figure, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim([-50, 50])
ax.set_xlim([-50, 50])

colors_array = ['b', 'g', 'r', 'c', 'm', 'y'] * 2


def plot_result(i):
    """
    plot from the locations vehicles start and end their service
    the vehicle_solution is a chromsome as the shown in each part

    Parameters:
    ------------
    data_array : array
        NOTE: THIS IS A `GLOBAL VARIABLE`

        The customers visited and the depository data 
        there should be four value in each index
         (0) is the symbol or the customer number
         (1) is the X location
         (2) is the Y location
         (3) is the vehicle number
    
    """
    ax.clear()
    data_arr = []

    data_x = []
    data_y = []
    data_text = []
    data_color = []

    for index in range(i+1):
        data_arr.append(DATA_TO_PLOT[index])

    for data in data_arr:
        # if re.match('\(\d+\)',data[0]):
        if data[0] == '(1)' or data[0] == '(2)' or data[0] == '(3)' or data[0] == '(4)' or data[0] == '(5)' or data[0] == '(6)':
            ax.scatter(data[1], data[2], marker='+', c='k', label=f'Depot: {data[0]}', zorder=1)
            data_x.append(data[1])
            data_y.append(data[2])
        else:
            
            color = colors_array[data[3]]
            data_color.append(color)
            data_x.append(data[1])
            data_y.append(data[2])
            data_text.append(f'{data[0]}')

            # ax.plot(data[1], data[2], label=f'vehicle number:{data[3]}', marker='^', linestyle='--', alpha=0.5, zorder=-1, c=color)
            ax.text(data[1], data[2], s=f'{data[0]}')
        ax.plot(data_x, data_y, linestyle='--', alpha=0.3, zorder=-1, color='c')
        
        # ax.legend()

            
def get_results(chromosome, depot_locations_dict, dataset):
    """
    update the global DATA_TO_PLOT array for animations
    """

    for vehicle_number, vehicle in enumerate(chromosome.split('|')):
        ## data will be saved to be shown in this array
        ## there would be four value in each index
        ## (0) is the symbol or the customer number
        ## (1) is the X location
        ## (2) is the Y location
        ## (3) is the vehicle number
        for idx in range(3, len(vehicle) + 1, 3):
            symbol = vehicle[idx-3: idx]
            ## if it was a repository
            if re.match('\(\d+\)', symbol):
                packed_data = (symbol, depot_locations_dict[symbol][0], depot_locations_dict[symbol][1], None)
                DATA_TO_PLOT.append(packed_data)
            ## or if it was a customer
            else:
                customer_num = (int(symbol) - 100)
                customer = dataset[dataset.number == customer_num]
                loc_x, loc_y = customer.x.values[0], customer.y.values[0]
                packed_data = (customer_num, loc_x, loc_y, vehicle_number)
                DATA_TO_PLOT.append(packed_data)
    

def animate(results_name):
    """
    animate the result chromosome of the evolutionary algorithm
    plot_function is called to multiple time to plot the chromsome result 
    """
    ani = FuncAnimation(figure, plot_result, frames=len(DATA_TO_PLOT), interval=80, repeat=False)
    ani.save(os.path.join('results',f'{results_name}.gif'), dpi=300, writer=PillowWriter(fps=25))

    # plt.show()

if __name__ == '__main__':
    ## dataset name and the directory such as `p1_data.txt`
    dataset_loc = None
    chromosome_loc = None
    depots_location = None
    result_name = None
    try:
        dataset_loc = sys.argv[1] 
        chromosome_loc = sys.argv[2]
        depots_location = sys.argv[3]
        result_name = sys.argv[4]
    except IndexError as error:
        print(f"not enough parameters, error: {error}")
        sys.exit(0)

    chromsome = None

    with open(chromosome_loc, mode='r') as chromsome_file:
        chromsome = chromsome_file.read()
    with open(depots_location, mode='r') as depot_file:
        depots_location = json.load(depot_file)
    
    dataset = pd.read_csv(dataset_loc, sep=' ')

    ## get and save all the results in a global array
    get_results(chromsome, depots_location, dataset)
    # sys.exit(0)
    animate(result_name)
    