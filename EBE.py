import os
import numpy as np
import pandas as pd
import logging
import hic
import plasma
import jets
import plasma_interaction as pi
import config
import utilities
from utilities import resultsFrame, tempDir


# Exits temporary directory, saves dataframe to pickle, and dumps all temporary data.
def safe_exit(resultsDataFrame, temp_dir, filename, identifier):
    # Save dataframe
    # Note that we exit the directory first in order to retain a valid current working directory when cleaned up.
    # This prevents os.get_cwd() from throwing an error.
    os.chdir('..')
    if os.path.exists('/results/{}'.format(identifier)):
        pass
    else:
        logging.info('Making results directory...')
        os.mkdir('/results/{}'.format(identifier))

    logging.info('Saving progress...')
    logging.info(resultsDataFrame)
    resultsDataFrame.to_pickle(os.getcwd() + '/results/' + str(filename) + '.pkl')  # Save dataframe to pickle

    # Clear everything in the temporary directory and delete it, thereby deleting all event files.
    logging.info('Cleaning temporary directory...')
    logging.debug('This dumps all of the event data!')
    try:
        temp_dir.cleanup()
    except TypeError:
        pass


# Function to generate a new HIC event and sample config.NUM_SAMPLES jets in it.
def run_event(eventNo):

    trentoDataframe = hic.generate_event()

    seed = trentoDataframe.iloc[0]['seed']

    results = resultsFrame()

    ########################
    # Objectify Background #
    ########################
    # Create plasma object
    # Open the hydro file and create file object for manipulation.
    plasmaFilePath = 'viscous_14_moments_evo.dat'
    current_file = plasma.osu_hydro_file(file_path=plasmaFilePath, event_name='seed: {}'.format(seed))

    # Create event object
    # This asks the hydro file object to interpolate the relevant functions and pass them on to the plasma object.
    current_event = plasma.plasma_event(event=current_file)

    ################
    # Jet Analysis #
    ################
    # Oversample the background with jets
    for jetNo in range(0, config.EBE.NUM_SAMPLES):

        # Select jet production point
        if not config.mode.VARY_POINT:
            x0 = 0
            y0 = 0
        else:
            newPoint = hic.generate_jet_point(current_event)
            x0, y0 = newPoint[0], newPoint[1]

        # Select jet production angle
        theta0 = np.random.uniform(0, 2 * np.pi)

        # Generate jet object
        current_jet = jets.jet(x0=x0, y0=y0,
                               theta0=theta0, event=current_event, energy=config.jet.JET_ENERGY)

        # Find phase change times along jet trajectory
        logging.info('Calculating phase change times...')
        t_hrg = False
        t_unhydro = False
        hrg_time_total = 0
        unhydro_time_total = 0
        for t in np.arange(start=current_event.t0, stop=current_event.tf, step=config.transport.TIME_STEP):
            current_temp = current_jet.temp(current_event, time=t)
            if current_temp < config.transport.hydro.T_HRG:
                if bool(t_hrg) is False:
                    t_hrg = t
                if current_temp > config.transport.hydro.T_UNHYDRO:
                    hrg_time_total += config.transport.TIME_STEP
            if current_temp < config.transport.hydro.T_UNHYDRO:
                if bool(t_unhydro) is False:
                    t_unhydro = t
                unhydro_time_total += config.transport.TIME_STEP
        plasma_time_total = (current_event.tf - current_event.t0) - hrg_time_total - unhydro_time_total

        # Sample for shower correction
        # Currently just zero
        logging.info('Sampling shower correction distribution...')
        logging.debug('No shower correction for now!')
        shower_correction = 0

        # Calculate momentPlasma
        logging.info('Calculating unhydrodynamic moment:')
        momentUnhydro, momentUnhydroErr = pi.moment_integral(event=current_event, jet=current_jet, k=config.moment.K,
                                                             minTemp=0, maxTemp=config.transport.hydro.T_UNHYDRO)
        logging.info('Calculating hadron gas moment:')
        momentHrg, momentHrgErr = pi.moment_integral(event=current_event, jet=current_jet, k=config.moment.K,
                                                     minTemp=config.transport.hydro.T_UNHYDRO,
                                                     maxTemp=config.transport.hydro.T_HRG)
        logging.info('Calculating plasma moment:')
        momentPlasma, momentPlasmaErr = pi.moment_integral(event=current_event, jet=current_jet, k=config.moment.K,
                                                           minTemp=config.transport.hydro.T_HRG)

        # Calculate deflection angle
        # Basic trig with k=0.
        if config.moment.K == 0:
            deflection_angle_plasma = np.arctan((momentPlasma / current_jet.energy)) * (180 / np.pi)
            deflection_angle_plasma_error = np.arctan((momentPlasmaErr / current_jet.energy)) * (180 / np.pi)
            deflection_angle_hrg = np.arctan((momentHrg / current_jet.energy)) * (180 / np.pi)
            deflection_angle_hrg_error = np.arctan((momentHrgErr / current_jet.energy)) * (180 / np.pi)
            deflection_angle_unhydro = np.arctan((momentUnhydro / current_jet.energy)) * (180 / np.pi)
            deflection_angle_unhydro_error = np.arctan((momentUnhydroErr / current_jet.energy)) * (180 / np.pi)
        else:
            deflection_angle_plasma = None
            deflection_angle_plasma_error = None
            deflection_angle_hrg = None
            deflection_angle_hrg_error = None
            deflection_angle_unhydro = None
            deflection_angle_unhydro_error = None

        # Create momentPlasma results dataframe
        momentDataframe = pd.DataFrame(
            {
                "eventNo": [eventNo],
                "jetNo": [jetNo],
                "pT_plasma": [momentPlasma],
                "pT_plasma_error": [momentPlasmaErr],
                "pT_hrg": [momentHrg],
                "pT_hrg_error": [momentHrgErr],
                "pT_unhydro": [momentUnhydro],
                "pT_unhydro_error": [momentUnhydroErr],
                "k_moment": [config.moment.K],
                "deflection_angle_plasma": [deflection_angle_plasma],
                "deflection_angle_plasma_error": [deflection_angle_plasma_error],
                "deflection_angle_hrg": [deflection_angle_hrg],
                "deflection_angle_hrg_error": [deflection_angle_hrg_error],
                "deflection_angle_unhydro": [deflection_angle_unhydro],
                "deflection_angle_unhydro_error": [deflection_angle_unhydro_error],
                "shower_correction": [shower_correction],
                "X0": [x0],
                "Y0": [y0],
                "theta0": [theta0],
                "t_unhydro": [t_unhydro],
                "t_hrg": [t_hrg],
                "time_total_plasma": [plasma_time_total],
                "time_total_hrg": [hrg_time_total],
                "time_total_unhydro": [unhydro_time_total],
                "initial_time": [current_event.t0],
                "final_time": [current_event.tf],
            }
        )

        # Merge the trento and momentPlasma dataframes
        currentResultDataframe = pd.concat([momentDataframe, trentoDataframe], axis=1)

        results = results.append(currentResultDataframe)

        # Declare jet complete
        logging.info('Jet ' + str(jetNo) + ' Complete')

    return results


################
# Main Program #
################

# Instantiate counters
part = 0
eventNo = 0

# Set up results frame and filename.
temp_dir = None  # Instantiates object for interrupt before temp_dir created.
results = resultsFrame()
identifierString = str(int(np.random.uniform(0, 10000000)))
resultsFilename = 'results' + identifierString + 'p' + str(part)

# Create log file
logging.basicConfig(filename=os.getcwd() + '/results/{}/log_{}.log'.format(identifierString, identifierString),
                    encoding='utf-8', level=logging.DEBUG)

# Copy config file to results directory, tagged with identifier
utilities.run_cmd(*['cp', 'config.yml', os.getcwd()
                    + '/results/{}/config_{}.yml'.format(identifierString, identifierString)], quiet=True)

# Run event loop
try:
    while config.EBE.NUM_EVENTS == 0 or eventNo < config.EBE.NUM_EVENTS:
        # Create and move to temporary directory
        temp_dir = tempDir()

        # Generate a new HIC event and samples config.NUM_SAMPLES jets in it
        # Append returned dataframe to current dataframe
        results = results.append(run_event(eventNo=eventNo))

        # Exits directory, saves all current data, and dumps temporary files.
        safe_exit(resultsDataFrame=results, temp_dir=temp_dir, filename=resultsFilename, identifier=identifierString)

        if len(results) > 10000:
            part += 1
            resultsFilename = 'results' + identifierString + 'p' + str(part)
            results = resultsFrame()

        eventNo += 1

except KeyboardInterrupt:
    logging.warning('Interrupted!')
    logging.info('Cleaning up...')

    # Clean up and get everything sorted
    safe_exit(resultsDataFrame=results, temp_dir=temp_dir, filename=resultsFilename, identifier=identifierString)

except hic.StopEvent:
    logging.warning('HIC event error.')
    logging.info('Cleaning up...')

    # Clean up and get everything sorted
    safe_exit(resultsDataFrame=results, temp_dir=temp_dir, filename=resultsFilename, identifier=identifierString)

logging.info('Results identifier: {}'.format(identifierString))
logging.info('Please have an excellent day. :)')
