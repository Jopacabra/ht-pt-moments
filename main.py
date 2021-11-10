# Import system packages
import sys
import csv

# Import public packages
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

# Import project modules
sys.path.append(".")
import grid_reader as gr
import medium_integral as mi

# Build array of interpolated backgrounds

# Initialize empty array for interpolating functions
temp_func_array = np.array([])
vel_x_func_array = np.array([])
vel_y_func_array = np.array([])

# Temporary file array for testing
folder_list = np.array(['viscous_14_moments_evo.dat'])

# Iterate through list of backgrounds and interpolate each one
# Append callable interpolating function to array of functions
i = 0
for file in folder_list:

    # read the grid file
    print('Reading grid data ... event ' + str(i))
    grid_data, grid_width, NT = gr.load_grid_file(file)

    # Interpolate temperatures
    print('Interpolating temp grid data ... event ' + str(i))
    temp_func = gr.interpolate_temp_grid(grid_data, grid_width, NT)
    temp_func_array = np.append(temp_func_array, temp_func)

    # Interpolate x velocities
    print('Interpolating x vel. grid data ... event ' + str(i))
    vel_x_func = gr.interpolate_x_vel_grid(grid_data, grid_width, NT)
    vel_x_func_array = np.append(vel_x_func_array, vel_x_func)

    # Interpolate y velocities
    print('Interpolating y vel. grid data ... event ' + str(i))
    vel_y_func = gr.interpolate_y_vel_grid(grid_data, grid_width, NT)
    vel_y_func_array = np.append(vel_y_func_array, vel_x_func)

    i += 1

# Count and print number of backgrounds loaded
NUM_BACKGROUNDS = len(temp_func_array)
print("Number of Backgrounds: " + str(NUM_BACKGROUNDS))

# Call particular temp function with:
# tempFuncArray[EVENT](np.array([TIME,XPOS,YPOS]))
# Here TIME, XPOS, and YPOS are in fm.

# Initialize result pandas dataframe
results = pd.DataFrame(
        {
            "JET_ID": [],
            "event_num": [],
            "X0": [],
            "Y0": [],
            "theta0": [],
            "pT_moment": [],
        }
    )

# Set moment and number of jets
N = 32  # Number of jets to produce
K = int(0)  # Which moment to calculate

# Preps file information and creates result file
result_id = np.random.randint(0, 999999)
print("Result ID number: " + str(result_id))
results.to_csv('results/results' + str(result_id) + '.csv', mode='a', index=False, header=True)

# Randomly choose N jet samples
# Each sample has: event_num, X0, Y0, theta0
# Adds each result to array of moments, with associated arrays of input parameters
print("Calculating moments for " + str(N) + ' jets...')
for id in range(N):
    print("Calculating moment for jet " + str(id) + '...')
    # Pick event
    event_num = np.random.randint(0, NUM_BACKGROUNDS)  # random integer on [0,numBackgrounds)

    # Generate random initial conditions
    X0 = 0  #np.random.uniform(-1, 1)
    Y0 = 0  #np.random.uniform(-1, 1)
    theta0 = (2*np.pi/(N))*(id+1)  #np.random.uniform(0, 2 * np.pi)

    # Calculate moment
    moment = mi.moment_integral(temp_func_array[event_num], vel_x_func_array[event_num],
                                vel_y_func_array[event_num], X0, Y0, theta0, K)

    # Write data to a new dataframe
    currentResults = pd.DataFrame(
            {
                "JET_ID": [id],
                "event_num": [event_num],
                "X0": [X0],
                "Y0": [Y0],
                "theta0": [theta0],
                "pT_moment": [moment],
            }
        )

    # Append current result step to dataframe
    results = results.append(currentResults)

    if (id != 0 and id % 100 == 0):
        # Write to csv
        print('Saving progress...')
        results.to_csv('results/results' + str(result_id) + '.csv', mode='a', index=False, header=False)

        # Recreate empty results dataframe
        column_names = [x for x in results.columns]
        results = pd.DataFrame(columns=column_names)
    elif id == N - 1:
        # Write to csv
        print('Saving results...')
        results.to_csv('results/results' + str(result_id) + '.csv', mode='a', index=False, header=False)
    else:
        pass

# Remind user of ID and quit
print("Results saved with ID number: " + str(result_id))
print("Done! Have a lovely day. :)")