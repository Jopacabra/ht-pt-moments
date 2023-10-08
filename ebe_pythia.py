import os
import numpy as np
import pandas as pd
import logging
import collision
import plasma
import jets
import config
import utilities
from utilities import tempDir
import timekeeper
import pythia


# Exits temporary directory, saves dataframe to parquet, and dumps all temporary data.
def safe_exit(resultsDataFrame, hadrons_df, temp_dir, filename, identifier, keep_event=False):
    # Save hydro event file
    if keep_event:
        logging.info('Saving event hydro data...')
        # Copy config file to results directory, tagged with identifier
        try:
            utilities.run_cmd(*['mv', 'viscous_14_moments_evo.dat',
                                results_path + '/hydro_grid_{}.dat'.format(identifierString, identifierString)],
                              quiet=False)
        except FileNotFoundError:
            logging.error('Failed to copy grid file -- file not found')

        try:
            utilities.run_cmd(*['mv', 'surface.dat',
                                results_path + '/hydro_surface_{}.dat'.format(identifierString, identifierString)],
                              quiet=False)
        except FileNotFoundError:
            logging.error('Failed to copy surface file -- file not found')

    # Save the dataframe into the identified results folder
    logging.info('Saving progress...')
    logging.debug(resultsDataFrame)
    resultsDataFrame.to_parquet(results_path + '/{}.parquet'.format(filename))  # Save dataframe to parquet
    hadrons_df.to_parquet(results_path + '/{}_hadrons.parquet'.format(filename))

    # Return to the directory in which we ran the script.
    os.chdir(home_path)

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
    hadrons = pd.DataFrame({})

    ###############################
    # Generate new event geometry #
    ###############################

    logging.info('Generating new event...')

    # Run event generation using config setttings
    # Note that we need write permissions in the working directory
    event_dataframe = collision.generate_event(working_dir=None)
    rmax = event_dataframe.iloc[0]['rmax']

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
        #########################
        # Create new scattering #
        #########################
        particles, weight = pythia.scattering()

        logging.info('- Scattering {} -'.format(jetNo))

        i = 0
        for index, particle in particles.iterrows():
            # Only do the things for the particle output
            particle_status = particle['status']
            if particle_status != 23:
                i += 1
                continue
            # Read jet properties
            chosen_e = particle['pt']
            chosen_weight = weight
            particle_pid = particle['id']
            if particle_pid == 21:
                chosen_pilot = 'g'
            elif particle_pid == 1:
                chosen_pilot = 'd'
            elif particle_pid == -1:
                chosen_pilot = 'dbar'
            elif particle_pid == 2:
                chosen_pilot = 'u'
            elif particle_pid == -2:
                chosen_pilot = 'ubar'
            elif particle_pid == 3:
                chosen_pilot = 's'
            elif particle_pid == -3:
                chosen_pilot = 'sbar'

            # Select jet production point
            if not config.mode.VARY_POINT:
                x0 = 0
                y0 = 0
            else:
                newPoint = collision.generate_jet_point(event)
                x0, y0 = newPoint[0], newPoint[1]

            # Read jet production angle
            phi_0 = np.arctan2(particle['py'], particle['px']) + np.pi

            # Yell about your selected jet
            logging.info('Pilot parton: {}, pT: {} GeV'.format(chosen_pilot, chosen_e))

            el_model = 'SGLV'
            for case in [1]:
                # Determine case details
                if case == 0:
                    drift = False
                    fg = False
                    grad = False
                    el = True
                elif case == 1:
                    drift = True
                    fg = False
                    grad = False
                    el = True
                elif case == 2:
                    drift = True
                    fg = True
                    grad = False
                    el = True
                elif case == 3:
                    drift = True
                    fg = True
                    grad = True
                    el = True
                else:
                    drift = True
                    fg = True
                    grad = False
                    el = True

                # Log jet number and case description
                logging.info('Running Jet {}, case {}'.format(str(jetNo), case))
                logging.info('Energy Loss: {}, Vel Drift: {}, Grad Drift: {}'.format(el, drift, grad))

                # Create the jet object
                jet = jets.jet(x_0=x0, y_0=y0, phi_0=phi_0, p_T0=chosen_e, tag=jet_tag, no=jetNo, part=chosen_pilot,
                               weight=chosen_weight)

                # Run the time loop
                jet_dataframe, jet_xarray = timekeeper.time_loop(event=event, jet=jet, drift=drift, el=el, grad=grad, fg=fg,
                                                                 el_model=el_model)

                # Save the xarray trajectory file
                # Note we are currently in a temp directory... Save record in directory above.
                if config.jet.RECORD:
                    jet_xarray.to_netcdf('../{}_record.nc'.format(jet_tag))

                # Merge the event and jet dataframe lines
                current_result_dataframe = pd.concat([jet_dataframe, event_dataframe], axis=1)

                # Append the total dataframe to the results dataframe
                results = pd.concat([results, current_result_dataframe], axis=0)

                # Save jet pair
                if i == 4:
                    jet1 = jet
                elif i == 5:
                    jet2 = jet

                i += 1

        # Hadronize jet pair
        current_hadrons = pythia.fragment(jet1=jet1, jet2=jet2, process_dataframe=particles, weight=chosen_weight)
        hadrons = pd.concat([hadrons, current_hadrons], axis=0)

        # Declare jet complete
        logging.info('- Jet ' + str(jetNo) + ' Complete -')

    return results, hadrons


################
# Main Program #
################

# Instantiate counters
part = 0
eventNo = 0

# Set up results frame and filename.
temp_dir = None  # Instantiates object for interrupt before temp_dir created.
results = pd.DataFrame({})
hadrons = pd.DataFrame({})
identifierString = str(int(np.random.uniform(0, 10000000)))
resultsFilename = 'results' + identifierString + 'p' + str(part)

# Set running location as current directory - whatever the pwd was when running the script
project_path = os.path.dirname(os.path.realpath(__file__))  # Gets directory the EBE.py script is located in
home_path = os.getcwd()  # Gets working directory when script was run - results directory will be placed here
results_path = home_path + '/results/{}'.format(identifierString)  # Absolute path of dir where results files will live

# Make results directory
os.makedirs(results_path, exist_ok=True)

# Create log file & configure logging to be handled into the file AND stdout
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(results_path + '/log_{}.log'.format(identifierString)),
        logging.StreamHandler()
    ]
)

# Copy config file to results directory, tagged with identifier
logging.info('Copying config.yml to results...')
utilities.run_cmd(*['cp', project_path + 'config.yml', results_path + '/config_{}.yml'.format(identifierString)],
                  quiet=True)

# Run event loop
try:
    while config.EBE.NUM_EVENTS == 0 or eventNo < config.EBE.NUM_EVENTS:
        # Create and move to temporary directory
        temp_dir = tempDir(location=results_path)
        print(os.getcwd())

        # Generate a new HIC event and sample config.NUM_SAMPLES jets in it
        # Append returned dataframe to current dataframe
        event_results, event_hadrons = run_event(eventNo=int(identifierString))
        results = pd.concat([results, event_results], axis=0)
        hadrons = pd.concat([results, event_hadrons], axis=0)

        # Exits directory, saves all current data, and dumps temporary files.
        safe_exit(resultsDataFrame=results, hadrons_df=hadrons, temp_dir=temp_dir, filename=resultsFilename, identifier=identifierString,
                  keep_event=config.mode.KEEP_EVENT)

        if len(results) > 10000:
            part += 1
            resultsFilename = 'results' + identifierString + 'p' + str(part)
            results = pd.DataFrame({})
            hadronssFilename = 'results' + identifierString + 'p' + str(part)
            hadronss = pd.DataFrame({})

        eventNo += 1

except KeyboardInterrupt as error:
    logging.exception('Interrupted!!!: {}'.format(str(error)))
    logging.info('Cleaning up...')

    # Clean up and get everything sorted
    safe_exit(resultsDataFrame=results, hadrons_df=hadrons, temp_dir=temp_dir, filename=resultsFilename, identifier=identifierString,
              keep_event=config.mode.KEEP_EVENT)

except collision.StopEvent as error:
    logging.exception('HIC event error: {}'.format(str(error)))
    logging.info('Cleaning up...')

    # Clean up and get everything sorted
    safe_exit(resultsDataFrame=results, hadrons_df=hadrons, temp_dir=temp_dir, filename=resultsFilename, identifier=identifierString,
              keep_event=config.mode.KEEP_EVENT)

except MemoryError as error:
    logging.exception('Memory error: {}'.format(str(error)))
    logging.info('Cleaning up...')

    # Clean up and get everything sorted
    safe_exit(resultsDataFrame=results, hadrons_df=hadrons, temp_dir=temp_dir, filename=resultsFilename, identifier=identifierString,
              keep_event=config.mode.KEEP_EVENT)

except BaseException as error:
    logging.exception('Unhandled error: {}'.format(str(error)))
    logging.info('Attempting to clean up...')

    # Clean up and get everything sorted
    safe_exit(resultsDataFrame=results, hadrons_df=hadrons, temp_dir=temp_dir, filename=resultsFilename, identifier=identifierString,
              keep_event=config.mode.KEEP_EVENT)

logging.info('Results identifier: {}'.format(identifierString))
logging.info('Successful clean exit!')
logging.info('Please have an excellent day. :)')
