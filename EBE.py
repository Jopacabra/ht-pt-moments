import os
import numpy as np
import pandas as pd
import xarray as xr
import logging
import hic
import plasma
import jets
import plasma_interaction as pi
import config
import utilities
from utilities import tempDir
import timekeeper


# Exits temporary directory, saves dataframe to pickle, and dumps all temporary data.
def safe_exit(resultsDataFrame, temp_dir, filename, identifier, keep_event=False):
    # Save hydro event file
    if keep_event:
        logging.info('Saving event hydro data...')
        # Copy config file to results directory, tagged with identifier
        utilities.run_cmd(*['mv', 'viscous_14_moments_evo.dat',
                            '../hydro_grid_{}.dat'.format(identifierString, identifierString)], quiet=False)

    # Note that we exit the directory first in order to retain a valid current working directory when cleaned up.
    # This prevents os.get_cwd() from throwing an error.

    os.chdir('..')

    # Save the dataframe into the identified results folder
    logging.info('Saving progress...')
    logging.debug(resultsDataFrame)
    resultsDataFrame.to_pickle(os.getcwd() + '/{}.pkl'.format(filename))  # Save dataframe to pickle

    if keep_event:
        logging.info('Saving event hydro data...')
        # Copy config file to results directory, tagged with identifier
        utilities.run_cmd(*['mv', 'viscous_14_moments_evo.dat', os.getcwd()
                            + '/hydro_grid_{}.dat'.format(identifierString, identifierString)], quiet=True)

    # Return to the project root.
    os.chdir('..')
    os.chdir('..')

    # Clear everything in the temporary directory and delete it, thereby deleting all event files.
    logging.info('Cleaning temporary directory...')
    logging.debug('This dumps all of the event data!')
    try:
        temp_dir.cleanup()
    except TypeError:
        pass


# Function to generate a new HIC event and sample config.NUM_SAMPLES jets in it.
def run_event(eventNo):

    # Generate empty results frame
    results = pd.DataFrame({})

    ###############################
    # Generate new event geometry #
    ###############################

    logging.info('Generating new event...')

    # Run event generation using config setttings
    event_dataframe, rmax = hic.generate_event(get_rmax=True)

    # Record seed selected
    seed = event_dataframe.iloc[0]['seed']

    # Open the hydro file and create file object for manipulation.
    plasmaFilePath = 'viscous_14_moments_evo.dat'
    file = plasma.osu_hydro_file(file_path=plasmaFilePath, event_name='seed: {}'.format(seed))

    # Create event object
    # This asks the hydro file object to interpolate the relevant functions and pass them on to the plasma object.
    event = plasma.plasma_event(event=file, name=eventNo, rmax=rmax)

    ################
    # Jet Analysis #
    ################
    # Oversample the background with jets
    for jetNo in range(0, config.EBE.NUM_SAMPLES):
        # Create unique jet tag
        jet_tag = str(int(np.random.uniform(0, 1000000000000)))
        ##################
        # Create new jet #
        ##################
        logging.info('- Jet {} -'.format(jetNo))
        # Select jet production point
        if not config.mode.VARY_POINT:
            x0 = 0
            y0 = 0
        else:
            newPoint = hic.generate_jet_point(event)
            x0, y0 = newPoint[0], newPoint[1]

        # Select jet production angle
        phi_0 = np.random.uniform(0, 2 * np.pi)

        # Select jet energy
        if config.jet.E_FLUCT:
            chosen_pilot, chosen_e = hic.jet_sample_LHC(cent=None)
        else:
            chosen_pilot = 'g'
            chosen_e = config.jet.JET_ENERGY

        logging.info('Pilot parton: {}, pT: {} GeV'.format(chosen_pilot, chosen_e))

        for case in ['db', 'd', 'b', 'd/2b', '2db']:
            logging.info('Running Jet {}, Case {}'.format(str(jetNo), case))
            # Create the jet object
            jet = jets.jet(x_0=x0, y_0=y0, phi_0=phi_0, p_T0=chosen_e, tag=jet_tag, no=jetNo, part=chosen_pilot)

            if case == 'db':
                g_drift = 1
                drift = True
                bbmg = True
            elif case == 'd':
                g_drift = 1
                drift = True
                bbmg = False
            elif case == 'b':
                g_drift = 0
                drift = False
                bbmg = True
            elif case == '2db':
                g_drift = 2
                drift = True
                bbmg = True
            elif case == 'd/2b':
                g_drift = 1/2
                drift = True
                bbmg = True
            else:
                g_drift = 0
                g_BBMG = 0
                drift = False
                bbmg = False
            # Run the time loop
            jet_dataframe, jet_xarray = timekeeper.time_loop(event=event, jet=jet, drift=drift, bbmg=bbmg,
                                                             g_drift=g_drift)

            # Merge the event and jet dataframe lines
            current_result_dataframe = pd.concat([jet_dataframe, event_dataframe], axis=1)

            # Append the total dataframe to the results dataframe
            results = pd.concat([results, current_result_dataframe], axis=0)

        # Declare jet complete
        logging.info('- Jet ' + str(jetNo) + ' Complete -')

    return results


################
# Main Program #
################

# Instantiate counters
part = 0
eventNo = 0

# Set up results frame and filename.
temp_dir = None  # Instantiates object for interrupt before temp_dir created.
results = pd.DataFrame({})
identifierString = str(int(np.random.uniform(0, 10000000)))
resultsFilename = 'results' + identifierString + 'p' + str(part)

# Make results directory
if os.path.exists(os.getcwd() + '/results/{}'.format(identifierString)):
    pass
else:
    print('Making results directory...')
    os.mkdir(os.getcwd() + '/results/{}'.format(identifierString))

# Create log file & configure logging to be handled into the file AND stdout
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.getcwd() + '/results/{}/log_{}.log'.format(identifierString, identifierString)),
        logging.StreamHandler()
    ]
)

# Copy config file to results directory, tagged with identifier
utilities.run_cmd(*['cp', 'config.yml', os.getcwd()
                    + '/results/{}/config_{}.yml'.format(identifierString, identifierString)], quiet=True)

# Run event loop
try:
    while config.EBE.NUM_EVENTS == 0 or eventNo < config.EBE.NUM_EVENTS:
        # Create and move to temporary directory
        temp_dir = tempDir(location=os.getcwd() + '/results/{}'.format(identifierString))

        # Generate a new HIC event and samples config.NUM_SAMPLES jets in it
        # Append returned dataframe to current dataframe
        results = results.append(run_event(eventNo=int(identifierString)))

        # Exits directory, saves all current data, and dumps temporary files.
        safe_exit(resultsDataFrame=results, temp_dir=temp_dir, filename=resultsFilename, identifier=identifierString,
                  keep_event=config.mode.KEEP_EVENT)

        if len(results) > 10000:
            part += 1
            resultsFilename = 'results' + identifierString + 'p' + str(part)
            results = pd.DataFrame({})

        eventNo += 1

except KeyboardInterrupt:
    logging.warning('Interrupted!')
    logging.info('Cleaning up...')

    # Clean up and get everything sorted
    safe_exit(resultsDataFrame=results, temp_dir=temp_dir, filename=resultsFilename, identifier=identifierString,
              keep_event=config.mode.KEEP_EVENT)

except hic.StopEvent:
    logging.warning('HIC event error.')
    logging.info('Cleaning up...')

    # Clean up and get everything sorted
    safe_exit(resultsDataFrame=results, temp_dir=temp_dir, filename=resultsFilename, identifier=identifierString,
              keep_event=config.mode.KEEP_EVENT)

logging.info('Results identifier: {}'.format(identifierString))
logging.info('Please have an excellent day. :)')
