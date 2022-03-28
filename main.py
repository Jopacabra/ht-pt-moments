import os
import numpy as np
import pandas as pd
import plasma as gr
import hard_scattering as hs
import plasma_interaction as mi
import jets
import config

# Build array of interpolated backgrounds

# Initialize empty array for plasma background objects
event_array = np.array([])

# Build array of files in directory
hydroResultsPath = 'backgrounds/'
file_list = os.listdir(path=hydroResultsPath)

# Iterate through list of backgrounds and interpolate each one.
# Append callable interpolating function to array of functions.
i = 0
for file_name in file_list:
    if file_name == 'README.md':
        continue

    # Specify the current file.
    current_file_path = hydroResultsPath + str(file_name)

    # Open the hydro file and create file object for manipulation.
    current_file = gr.osu_hydro_file(file_path=current_file_path, event_name=file_name)

    # Create event object and append to event object array.
    # This asks the hydro file object to interpolate the relevant functions
    # and pass them on to the plasma object.
    new_event = np.array([gr.plasma_event(event=current_file)])
    event_array = np.append(event_array, new_event)

    i += 1

# Count and print number of events loaded
NUM_EVENTS = len(event_array)
print("Number of Backgrounds: " + str(NUM_EVENTS))


# Call particular temp function with:
# tempFuncArray[EVENT](np.array([TIME,XPOS,YPOS]))
# Here TIME, XPOS, and YPOS are in fm.

# Initialize result pandas dataframe
emptyFrame = pd.DataFrame(
    {
        "JET_ID": [],
        "event_num": [],
        "event_name": [],
        "X0": [],
        "Y0": [],
        "theta0": [],
        "pT_moment": [],
        "pT_moment_error": [],
        "deflection_angle": [],
        "deflection_angle_error": [],
    }
)

results = emptyFrame


#
def WriteDataLine(ID, event_num, event_name, X0, Y0, theta0, moment, momentErr, defAngle, defAngleErr):
    dataframe = pd.DataFrame(
        {
            "JET_ID": [ID],
            "event_num": [event_num],
            "event_name": [event_name],
            "X0": [X0],
            "Y0": [Y0],
            "theta0": [theta0],
            "pT_moment": [moment],
            "pT_moment_error": [momentErr],
            "deflection_angle": [defAngle],
            "deflection_angle_error": [defAngleErr],
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

"""
# Operation mode for scanning each event. Old and deprecated, but it works if you wanna run a test case.
if RANDOM_EVENT == False:
    for event_num in range(NUM_EVENTS):
        for ID in range(N):
            print("Calculating moment for jet " + str(ID) + ' for event ' + str(event_num) + '...')
            # All events cycled through with complete angular profile.

            # Create jet object at [0,0] with theta0 the current incremented value.
            current_theta0 = (2 * np.pi / (N)) * (ID + 1)  # np.random.uniform(0, 2 * np.pi)
            current_jet = jets.jet(X0=0, Y0=0, theta0=current_theta0, event=event_array[event_num])
            

            # Calculate moment
            moment = mi.moment_integral(event=event_array[event_num], jet=current_jet, k=0)

            # Write data to a new dataframe
            currentResults = WriteDataLine(ID, event_num, current_jet.x0, current_jet.y0, current_jet.theta0, moment[0], moment[1])

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
"""


# Operation mode for random event selection.
for ID in range(config.N):

    # Randomly select event
    event_num = np.random.randint(0, NUM_EVENTS)  # random integer on [0,numBackgrounds)
    current_event = event_array[event_num]

    # Select jet production point
    if config.VARY_POINT:
        newPoint = hs.generate_jet_point(current_event)
        current_x0, current_y0 = newPoint[0], newPoint[1]
    elif not config.VARY_POINT:
        current_x0 = 0
        current_y0 = 0
    else:
        print("Configuration error. All jet prod points = (0,0).")
        current_x0 = 0
        current_y0 = 0

    # Select jet production angle
    if config.RANDOM_ANGLE:
        current_theta0 = np.random.uniform(0, 2 * np.pi)
    elif not config.RANDOM_ANGLE:
        current_theta0 = (2 * np.pi / (config.N)) * (ID + 1)  # np.random.uniform(0, 2 * np.pi)
    else:
        print("Configuration error. All jet angles random.")
        current_theta0 = np.random.uniform(0, 2 * np.pi)

    # Generate jet object
    current_jet = jets.jet(x0=current_x0, y0=current_y0,
                           theta0=current_theta0, event=current_event, energy=config.JET_ENERGY)

    # Calculate moment
    moment = mi.moment_integral(event=current_event, jet=current_jet, k=0)

    # Calculate deflection angle
    deflection_angle = np.arctan((moment[0]/current_jet.energy))*(180/np.pi)
    deflection_angle_error = np.arctan((moment[1]/current_jet.energy))*(180/np.pi)

    # Write data to a new dataframe
    currentResults = WriteDataLine(ID, event_num, current_event.name, current_jet.x0, current_jet.y0, current_jet.theta0,
                                   moment[0], moment[1], deflection_angle, deflection_angle_error)

    # Append current result step to dataframe
    results = results.append(currentResults)

    # Declare jet complete
    print('Jet ' + str(ID) + ' Complete')

    # Write to csv
    if ID % SAVE_EVERY == 0 or ID == config.N-1:

        # Save jets in results memory
        print('Saving progress...')
        results.to_csv('results/results' + str(result_id) + '.csv', mode='a', index=False, header=False)

        # Clear results memory
        # Recreate empty results dataframe for next jet
        column_names = [x for x in results.columns]
        results = pd.DataFrame(columns=column_names)

    # Declare if all jets complete
    if ID == config.N - 1:
        # Print completion Confirmation
        print(str(config.N) + ' Jets Completed')

# Remind user of ID and quit
print("Results saved with ID number: " + str(result_id))
print("Done! Have a lovely day. :)")