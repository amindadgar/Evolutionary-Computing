# Evolutionary-Computing
My Evolutionary-Computing course exercises.

## Homework #1
The first homework of this course was to find out how different recombination, selection, mutation, population size, and problem sizes of a problem can affect the performance of a genetic algorithm.

To Re-run the project you can simply do the followings
- ```pip install -r reqirements.txt```
- create a folder named Resutls in the root of the project.
- change the name of the config file in `__init__.py` if you want to run based on your config. Note that the keys in json file shouldn't change.
- ```python main.py 10``` to run with each config 10 times.
- to read the results just simply open and run the file ```show_results.ipynb```.

## Homework #2
The second homework was all about vehicle routing problem. We aimed to optimize 9 problems as the results are available in `HW2\main.ipynb`. Each problem's description is listed below:

- **PROBLEM No. 1**

    The goal is to minimize the distance gone, of the vehicles with the parameters below
    - There is 1 depot in the location `(-14, 9)`
    - There are `6` vehicles
    - Maximum Capacity is `70`
    - The locations of the customers and their demands are in `P1.txt` file
    - The distance each vehicle can go is not limited.

- **Problem No. 2**

    The goal is to maximize the customers recieved service with parametes below
    - There is 1 depot in the location `(0, 13)`
    - There are `4` vehicles
    - Maximum distace each vehicle can go is `200` Km
    - The locations of the customers and their demands are in `P2.txt` file
    - The capacity is not limited for vehicles

- **Problem No. 3**

    The goal is to maximize the demands of customers with parametes below
    - There is 1 depot in the location `(-17, -4)`
    - There are `4` vehicles
    - Maximum distace each vehicle can go is `200` Km
    - The locations of the customers and their demands are in `P3.txt` file
    - The capacity is not limited for vehicles

- **Problem No. 4**
    The goal is to minimize number of vehicles with parametes below
    - There is 1 depot in the location `(24, -7)`
    - There are `?` vehicles (optimization objective)
    - Maximum distace each vehicle can go is `200` Km
    - The locations of the customers and their demands are in `P4.txt` file
    - The capacity is not limited for vehicles

- **Problem No. 5**

    The goal is to minimize the distance gone, of the vehicles with the parameters below
    - There are 3 depot in the location `(31, 6)`, `(-31, 7)`, `(25, -10)`.
    - There are `11` vehicles
    - Maximum Capacity is `100`
    - The locations of the customers and their demands are in `P5.txt` file
    - The distance each vehicle can go is not limited.

- **Problem No. 6**
    The goal is to maximize the customers recieved service with parametes below
    - There are two depots in the location `(11, 36)`, `(19, -41)`
    - There are `7` vehicles
    - Maximum distace each vehicle can go is `250` Km
    - The locations of the customers and their demands are in `P6.txt` file
    - The capacity is not limited for vehicles

- **Problem No. 7**

    The goal is to maximize the demands of customers with parametes below
    - There are three depots in the locations `(44, -41)`, `(-24, -8)`, `(-33, 30)`, `(10, 43)`
    - There are `10` vehicles
    - Maximum distace each vehicle can go is `250` Km
    - The locations of the customers and their demands are in `P7.txt` file
    - The capacity is not limited for vehicles
- **Problem No. 8**

    Same as problem 4, but having multiple depot. Depots are: `(8, 17)`, `(31, -42)`, `(-6, -22)`, `(15, -10)`,`(-27, 43)`.

- **Problem No. 9**

    The goal is to minimum the distance gone as the Problems 2 and 5 but the difference is that the vehicle can go and back to another depot, and start from that depot

## Homework No. 3
This homework was a Neural Architecture Search (NAS) problem which optimizes a neural network's hyperparameters. An encoder part of transformer network was given and we were given the task of optimizing neuron count, attention function type and drop out probability of the transformer networks with layer count, output probability and model dimesionality.

The file `HW3\main.ipynb` contains the whole report for the implementation of the code and results. 

## Homework No. 4
This homework was a VRP problem as Homework Number 2 but there were multiple goals to be optimized at the same time. So multi-objective evolutionary algorithms such as NSGA-II and MOEA/D were used. for this homework we used pymoo library as an additional solution for our codes. The main file of running and outputs are available at `HW4\main.ipynb`.

