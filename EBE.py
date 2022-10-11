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
from utilities import tempDir


# Exits temporary directory, saves dataframe to pickle, and dumps all temporary data.
def safe_exit(resultsDataFrame, temp_dir, filename, identifier):
    # Save dataframe
    # Note that we exit the directory first in order to retain a valid current working directory when cleaned up.
    # This prevents os.get_cwd() from throwing an error.
    os.chdir('..')

    # Save the dataframe into the identified results folder
    logging.info('Saving progress...')
    logging.debug(resultsDataFrame)
    resultsDataFrame.to_pickle(os.getcwd() + '/{}.pkl'.format(filename))  # Save dataframe to pickle

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
    event = plasma.plasma_event(event=file)

    ################
    # Jet Analysis #
    ################
    # Oversample the background with jets
    for jetNo in range(0, config.EBE.NUM_SAMPLES):

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
            rng = np.random.default_rng()
            while True:
                chosen_e = rng.exponential(5)
                if chosen_e > config.jet.MIN_JET_ENERGY:
                    break
        else:
            chosen_e = config.jet.JET_ENERGY

        # Create the jet object
        jet = jets.jet(x_0=x0, y_0=y0, phi_0=phi_0, p_T0=chosen_e)


        #############
        # Time Loop #
        #############
        # Set loop parameters
        tau = config.jet.TAU  # dt for time loop in fm
        t = event.t0  # Set current time in fm to initial time

        # Initialize counters & values
        t_qgp = None
        t_hrg = None
        t_unhydro = None
        qgp_time_total = 0
        hrg_time_total = 0
        unhydro_time_total = 0
        maxT = 0
        q_bbmg_total = 0
        q_drift_total = 0

        # Initialize flags
        first = True
        qgp_first = True
        hrg_first = True
        unhydro_first = True
        phase = None

        # Initiate loop
        logging.info('Initiating time loop...')
        while True:
            #####################
            # Initial Step Only #
            #####################
            if first:
                # Good luck!
                first = False

            #########################
            # Set Current Step Data #
            #########################
            # Decide if we're in bounds of the grid
            if jet.x > event.xmax or jet.y > event.ymax:
                break
            elif jet.x < event.xmin or jet.y < event.ymin:
                break
            elif t > event.tf:
                break

            # For timekeeping in phases, we approximate all time in one step as in one phase
            temp = event.temp(jet.coords3(time=t))

            # Decide phase
            if temp > config.transport.hydro.T_HRG:
                phase = 'qgp'
            elif temp < config.transport.hydro.T_HRG and temp > config.transport.hydro.T_UNHYDRO:
                phase = 'hrg'
            elif temp < config.transport.hydro.T_UNHYDRO and temp > config.transport.hydro.T_END:
                phase = 'unh'

            ############################
            # Perform jet calculations #
            ############################

            if phase == 'qgp':
                # Calculate energy loss due to gluon exchange with the medium
                q_bbmg = tau * pi.energy_loss_integrand(event=event, jet=jet, time=t, tau=tau)

                # Calculate jet drift momentum transferred to jet
                q_drift = tau * pi.jet_drift_integrand(event=event, jet=jet, time=t)
            else:
                q_bbmg = 0
                q_drift = 0

            ###################
            # Data Accounting #
            ###################
            # Log momentum transfers
            q_bbmg_total += q_bbmg
            q_drift_total += q_drift

            # Check for max temperature
            if temp > maxT:
                maxT = temp

            # Decide phase for categorization & timekeeping
            if phase == 'qgp':
                if qgp_first:
                    t_qgp = t
                    qgp_first = False

                qgp_time_total += tau

            # Decide phase for categorization & timekeeping
            if phase == 'hrg':
                if hrg_first:
                    t_hrg = t
                    hrg_first = False

                hrg_time_total += tau

            if phase == 'unh':
                if unhydro_first:
                    t_unhydro = t
                    unhydro_first = False

                unhydro_time_total += tau

            #########################
            # Change Jet Parameters #
            #########################
            # Propagate jet position
            jet.prop(tau=tau)

            # Change jet momentum to reflect BBMG energy loss
            jet.add_q_par(q_par=q_bbmg)

            # Change jet momentum to reflect drift effects
            jet.add_q_perp(q_perp=q_drift)

            ###############
            # Timekeeping #
            ###############
            t += tau

        logging.info('Time loop complete...')
        # Create momentPlasma results dataframe
        jet_dataframe = pd.DataFrame(
            {
                "eventNo": [eventNo],
                "jetNo": [jetNo],
                "jet_pT": [jet.p_T0],
                "q_BBMG": [q_bbmg_total],
                "q_drift": [q_drift_total],
                "shower_correction": [jet.shower_correction],
                "X0": [x0],
                "Y0": [y0],
                "phi_0": [phi_0],
                "t_qgp": [t_qgp],
                "t_hrg": [t_hrg],
                "t_unhydro": [t_unhydro],
                "time_total_plasma": [qgp_time_total],
                "time_total_hrg": [hrg_time_total],
                "time_total_unhydro": [unhydro_time_total],
                "Tmax_jet": [maxT],
                "initial_time": [event.t0],
                "final_time": [event.tf],
                "dx": [config.transport.GRID_STEP],
                "dt": [config.transport.TIME_STEP],
                "tau": [config.jet.TAU],
                "rmax": [rmax],
                "Tmax_event": [event.max_temp()],
            }
        )

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
        results = results.append(run_event(eventNo=eventNo))

        # Exits directory, saves all current data, and dumps temporary files.
        safe_exit(resultsDataFrame=results, temp_dir=temp_dir, filename=resultsFilename, identifier=identifierString)

        if len(results) > 10000:
            part += 1
            resultsFilename = 'results' + identifierString + 'p' + str(part)
            results = pd.DataFrame({})

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
