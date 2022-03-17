import os
import numpy as np
import pandas as pd
import plasma_grid_reader as gr
import hard_scattering as hs
import plasma_interaction as mi
import yaml

# Read config file and parse settings
with open('config.yml', 'r') as ymlfile:
    # Note the usage of yaml.safe_load()
    # Using yaml.load() exposes the system to running any Python commands in the config file.
    # That is unnecessary risk!!!
    cfg = yaml.safe_load(ymlfile)

# Set scanning modes
RANDOM_EVENT = bool(cfg['mode']['RANDOM_EVENT'])  # Determines if we select a random event or iterate through each
RANDOM_ANGLE = bool(cfg['mode']['RANDOM_ANGLE'])  # Determines if we select a random angle or do a complete sweep
VARY_POINT = bool(cfg['mode']['VARY_POINT'])  # Determines if we vary the prod point or set it to (0,0)
WEIGHT_POINT = bool(cfg['mode'][
                        'WEIGHT_POINT'])  # Determines if we weight point selection by T^6 in event - Needs VARYPOINT.

# Set jet parameters
N = int(cfg['jet_parameters']['NUM_JETS'])  # Number of jets to produce
JET_ENERGY = int(
    cfg['jet_parameters']['JET_ENERGY'])  # Jet energy - Needs Testing

# Set moment parameters
K = int(cfg['moment_parameters']['MOMENT_K'])  # Which k-moment to calculate

# Set config settings

if RANDOM_ANGLE == False and VARY_POINT == False and RANDOM_EVENT == False:
    print('Calculating angular sweep of moments for central jets in each background...')
elif RANDOM_ANGLE == False and VARY_POINT == False and RANDOM_EVENT == True:
    print('Calculating angular sweep of moments for central jets in random backgrounds...')
elif RANDOM_ANGLE == True and VARY_POINT == True:
    print('Calculating a given number of random jets at random angles in random backgrounds from library...')
elif RANDOM_ANGLE == True and VARY_POINT == False:
    print('Calculating a given number of central jets at random angles in random backgrounds from library...')
elif RANDOM_ANGLE == False and VARY_POINT == True:
    print('Calculating angular sweep of random jets in random backgrounds from library...')
else:
    print('Input invalid mode!!!!!')
    print('RANDOM_ANGLE = ' + str(RANDOM_ANGLE))
    print('VARY_POINT = ' + str(VARY_POINT))
    print('RANDOM_EVENT = ' + str(RANDOM_EVENT))
    print('WEIGHT_POINT = ' + str(WEIGHT_POINT))

# Build array of interpolated backgrounds

# Initialize empty array for interpolating functions
temp_func_array = np.array([])
vel_x_func_array = np.array([])
vel_y_func_array = np.array([])

# Temporary file array for testing
folder_list = os.listdir(path='backgrounds/')

# Iterate through list of backgrounds and interpolate each one
# Append callable interpolating function to array of functions
i = 0
for file in folder_list:
    if file == 'README.md':
        continue
    # read the grid file
    print('Reading grid data ... event ' + str(i))
    grid_data, grid_width, NT = gr.load_grid_file(file)

    # Interpolate temperatures
    temp_func = gr.interpolate_temp_grid(grid_data, grid_width, NT)
    temp_func_array = np.append(temp_func_array, temp_func)

    # Interpolate x velocities
    vel_x_func = gr.interpolate_x_vel_grid(grid_data, grid_width, NT)
    vel_x_func_array = np.append(vel_x_func_array, vel_x_func)

    # Interpolate y velocities
    vel_y_func = gr.interpolate_y_vel_grid(grid_data, grid_width, NT)
    vel_y_func_array = np.append(vel_y_func_array, vel_x_func)

    i += 1

# Count and print number of backgrounds loaded
NUM_BACKGROUNDS = len(temp_func_array)
print("Number of Backgrounds: " + str(NUM_BACKGROUNDS))


def CalcMoments(event_num, X0, Y0, theta0, JET_E=10):
    # Calculate moment
    moment = mi.moment_integral(temp_func_array[event_num], vel_x_func_array[event_num],
                                vel_y_func_array[event_num], X0, Y0, theta0, K, JET_E=JET_E)
    return moment



# Call particular temp function with:
# tempFuncArray[EVENT](np.array([TIME,XPOS,YPOS]))
# Here TIME, XPOS, and YPOS are in fm.

# Initialize result pandas dataframe
emptyFrame = pd.DataFrame(
    {
        "JET_ID": [],
        "event_num": [],
        "X0": [],
        "Y0": [],
        "theta0": [],
        "pT_moment": [],
        "pT_moment_error": [],
    }
)

results = emptyFrame


#
def WriteDataLine(ID, event_num, X0, Y0, theta0, moment, momentErr):
    dataframe = pd.DataFrame(
        {
            "JET_ID": [ID],
            "event_num": [event_num],
            "X0": [X0],
            "Y0": [Y0],
            "theta0": [theta0],
            "pT_moment": [moment],
            "pT_moment_error": [momentErr],
        }
    )
    return dataframe


# Preps file information and creates result file
result_id = np.random.randint(0, 999999)
print("Result ID number: " + str(result_id))
results.to_csv('results/results' + str(result_id) + '.csv', mode='a', index=False, header=True)

#################
# Main Jet Loop #
#################
"""
Choose jets and backgrounds according to user specified mode (from config.yml file)

Each sample has: event_num, X0, Y0, theta0 and resulting pT_moment and pT_moment_error
Error is as of yet pretty useless. Getting huge numbers out of integration.

Adds each result to pandas dataframe of moments with associated input parameters
Saves results to csv every few jets and dumps active memory of the results.
"""
SAVE_EVERY = 10  # Set how many jets to calculate before each dump to file.

# Operation mode for scanning each event. Old and deprecated, but it works if you wanna run a test case.
if RANDOM_EVENT == False:
    for event_num in range(NUM_BACKGROUNDS):
        for ID in range(N):
            print("Calculating moment for jet " + str(ID) + ' for event ' + str(event_num) + '...')
            # All events cycled through with complete angular profile.
            # event_num = np.random.randint(0, NUM_BACKGROUNDS)  # random integer on [0,numBackgrounds)

            # Generate random initial conditions
            X0 = 0  # np.random.uniform(-1, 1)
            Y0 = 0  # np.random.uniform(-1, 1)
            theta0 = (2 * np.pi / (N)) * (ID + 1)  # np.random.uniform(0, 2 * np.pi)

            # Calculate moment
            moment = mi.moment_integral(temp_func_array[event_num], vel_x_func_array[event_num],
                                        vel_y_func_array[event_num], X0, Y0, theta0, K)

            # Write data to a new dataframe
            currentResults = WriteDataLine(ID, event_num, X0, Y0, theta0, moment[0], moment[1])

            # Append current result step to dataframe
            results = results.append(currentResults)

            # Write to csv
            print('Saving progress...')
            results.to_csv('results/results' + str(result_id) + '.csv', mode='a', index=False, header=False)

            # Recreate empty results dataframe for next jet
            column_names = [x for x in results.columns]
            results = pd.DataFrame(columns=column_names)
            # Print completion Confirmation
            print('Jet ' + str(ID) + ' Complete')

# Operation mode for random event selection.
elif RANDOM_EVENT == True:
    for ID in range(N):

        # Randomly select event
        event_num = np.random.randint(0, NUM_BACKGROUNDS)  # random integer on [0,numBackgrounds)

        # Select jet production point
        TARGETRADIUS = 1
        if VARY_POINT == True and WEIGHT_POINT == False:
            X0 = np.random.uniform(-TARGETRADIUS, TARGETRADIUS)
            Y0 = np.random.uniform(-TARGETRADIUS, TARGETRADIUS)
        elif VARY_POINT == True and WEIGHT_POINT == True:
            newPoint = hs.generate_jet_point(temp_func_array[event_num])
            X0, Y0 = newPoint[0], newPoint[1]
        elif VARY_POINT == False:
            X0 = 0
            Y0 = 0
        else:
            print("Configuration error. All jet prod points = (0,0).")
            X0 = 0
            Y0 = 0

        # Select jet production angle
        if RANDOM_ANGLE == True:
            theta0 = np.random.uniform(0, 2 * np.pi)
        elif RANDOM_ANGLE == False:
            theta0 = (2 * np.pi / (N)) * (ID + 1)  # np.random.uniform(0, 2 * np.pi)
        else:
            print("Configuration error. All jet angles = 0.")
            theta0 = 0

        # Calculate moment
        moment = CalcMoments(event_num, X0, Y0, theta0, JET_E=JET_ENERGY)

        # Write data to a new dataframe
        currentResults = WriteDataLine(ID, event_num, X0, Y0, theta0, moment[0], moment[1])

        # Append current result step to dataframe
        results = results.append(currentResults)

        # Declare jet complete
        print('Jet ' + str(ID) + ' Complete')

        # Write to csv
        if ID % SAVE_EVERY == 0 or ID == N-1:

            # Save jets in results memory
            print('Saving progress...')
            results.to_csv('results/results' + str(result_id) + '.csv', mode='a', index=False, header=False)

            # Clear results memory
            # Recreate empty results dataframe for next jet
            column_names = [x for x in results.columns]
            results = pd.DataFrame(columns=column_names)

        # Declare if all jets complete
        if ID == N - 1:
            # Print completion Confirmation
            print(str(N) + ' Jets Completed')

# Remind user of ID and quit
print("Results saved with ID number: " + str(result_id))
print("Done! Have a lovely day. :)")