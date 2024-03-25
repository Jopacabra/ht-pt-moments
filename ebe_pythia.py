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
import ff
import traceback

lund_string = False

# Function to downcast datatypes to minimum memory size for each column
def downcast_numerics(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':  # for integers
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:  # for floats.
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# Computes angular distance-ish quantity
def delta_R(phi1, phi2, y1, y2):
    dphi = np.abs(phi1 - phi2)
    if (dphi > np.pi):
        dphi = 2*np.pi - dphi
    drap = y1 - y2
    return np.sqrt(dphi * dphi + drap * drap)

# Exits temporary directory, saves dataframe to pickle, and dumps all temporary data.
def safe_exit(resultsDataFrame, temp_dir, filename, identifier, hadrons_df=None, keep_event=False):
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

    try:
        utilities.run_cmd(*['mv', '*.npy',
                            results_path + '/*.npy'],
                          quiet=False)
    except Exception as error:
        logging.info("Failed to copy UrQMD file: {}".format(type(error).__name__))  # An error occurred: NameError
        traceback.print_exc()

    # Save the dataframe into the identified results folder
    logging.info('Saving progress...')
    logging.info('Partons...')
    logging.debug(resultsDataFrame)
    logging.info('Converting datatypes to reasonable formats...')
    resultsDataFrame = resultsDataFrame.convert_dtypes()
    logging.info('Downcasting numeric values to minimum memory size...')
    resultsDataFrame = downcast_numerics(resultsDataFrame)

    if lund_string:
        logging.info('LS Hadrons...')
        logging.debug(hadrons_df)
        logging.info('Converting datatypes to reasonable formats...')
        hadrons_df = hadrons_df.convert_dtypes()
        logging.info('Downcasting numeric values to minimum memory size...')
        hadrons_df = downcast_numerics(hadrons_df)

    logging.info('Writing pickles...')
    resultsDataFrame.to_pickle(results_path + '/{}.pickle'.format(filename))  # Save dataframe to pickle
    if lund_string:
        hadrons_df.to_pickle(results_path + '/{}_hadrons.pickle'.format(filename))

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
    event_partons = pd.DataFrame({})
    event_hadrons = pd.DataFrame({})

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

    # Record event psi_2
    psi_2 = event_dataframe.iloc[0]['psi_2']

    # Open the hydro file and create file object for manipulation.
    plasmaFilePath = 'viscous_14_moments_evo.dat'
    file = plasma.osu_hydro_file(file_path=plasmaFilePath, event_name='seed: {}'.format(seed))

    # Create event object
    # This asks the hydro file object to interpolate the relevant functions and pass them on to the plasma object.
    event = plasma.plasma_event(event=file, name=eventNo, rmax=rmax)

    ################
    # Jet Analysis #
    ################
    # Select angular bin centers fixed at elliptic flow attractors and repulsors.
    # phi_res = np.pi/2
    # phi_bin_centers = np.arange(0, 2*np.pi, phi_res) + psi_2
    phi_rng = np.random.default_rng()
    num_phi = 11  # We select a prime number so this can't (?) influence v_n =/= v_{num_phi}

    # Oversample the background with jets
    for process_num in range(0, config.EBE.NUM_SAMPLES):
        # Create unique jet tag
        process_tag = int(np.random.uniform(0, 1000000000000))
        logging.info('- Jet Process {} Start -'.format(process_num))

        try:
            #########################
            # Create new scattering #
            #########################
            particles, weight = pythia.scattering()
            particle_tags = np.random.default_rng().uniform(0, 1000000000000, len(particles)).astype(int)
            process_partons = pd.DataFrame({})
            process_hadrons = pd.DataFrame({})

            # Select jet production point
            if not config.mode.VARY_POINT:
                x0 = 0
                y0 = 0
            else:
                newPoint = collision.generate_jet_point(event)
                x0, y0 = newPoint[0], newPoint[1]

            process_run = 0

            # Random azimuthal sampling
            # phi_values = phi_rng.uniform(0, 2 * np.pi, num_phi)

            # Uniform azimuthal sampling
            phi_values = np.linspace(start=0, stop=2*np.pi, num=num_phi, endpoint=False) + psi_2

            for phi_val in phi_values:
                # phi_val = np.mod(np.random.uniform(phi_center - phi_res/2, phi_center + phi_res/2), 2*np.pi)

                for case in [0, 1, 2]:
                    case_partons = pd.DataFrame({})
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

                    i = 0
                    jet_num = -1
                    for index, particle in particles.iterrows():
                        # Only do the things for the particle output
                        particle_status = particle['status']
                        particle_tag = int(particle_tags[i])
                        if particle_status != 23:
                            i += 1
                            continue
                        jet_num += 1
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

                        # Select jet angle from sample
                        if jet_num == 0:
                            phi_0 = phi_val
                        else:
                            phi_0 = np.mod(phi_val + np.pi, 2*np.pi)

                        # Read jet production angle
                        #phi_0 = np.arctan2(particle['py'], particle['px']) + np.pi

                        # Yell about your selected jet
                        logging.info('Pilot parton: {}, pT: {} GeV'.format(chosen_pilot, chosen_e))

                        el_model = 'SGLV'

                        # Log jet number and case description
                        logging.info('Running Jet {}, case {}'.format(str(process_num), case))
                        logging.info('Energy Loss: {}, Vel Drift: {}, Grad Drift: {}'.format(el, drift, grad))

                        # Create the jet object
                        jet = jets.jet(x_0=x0, y_0=y0, phi_0=phi_0, p_T0=chosen_e, tag=particle_tag, no=jet_num, part=chosen_pilot,
                                       weight=chosen_weight)

                        # Perform pp-level fragmentation
                        pp_frag_z = ff.frag(jet)

                        # Run the time loop
                        jet_dataframe, jet_xarray = timekeeper.time_loop(event=event, jet=jet, drift=drift, el=el, grad=grad, fg=fg,
                                                                         el_model=el_model)

                        # Save the xarray trajectory file
                        # Note we are currently in a temp directory... Save record in directory above.
                        if config.jet.RECORD:
                            jet_xarray.to_netcdf('../{}_record.nc'.format(process_tag))

                        # Add scattering process tag
                        jet_dataframe['process'] = process_tag

                        # Merge the event and jet dataframe lines
                        current_parton = pd.concat([jet_dataframe, event_dataframe], axis=1)

                        logging.info('FF Fragmentation')
                        # Perform ff fragmentation
                        frag_z = ff.frag(jet)
                        pion_pt = jet.p_T() * frag_z
                        pion_pt_0 = jet.p_T0 * pp_frag_z
                        current_parton['z'] = frag_z
                        current_parton['pp_z'] = pp_frag_z
                        current_parton['pion_pt_f'] = pion_pt
                        current_parton['pion_pt_0'] = pion_pt_0
                        current_parton['process_run'] = process_run

                        # Save jet pair
                        if i == 4:
                            jet1 = jet
                        elif i == 5:
                            jet2 = jet

                        # Append current partons to the case partons
                        case_partons = pd.concat([current_parton, case_partons], axis=0)

                        i += 1

                    if lund_string:
                        logging.info('Hadronizing...')
                        # Hadronize jet pair
                        scale = particles['scaleIn'].to_numpy()[-1]  # use last particle to set hard process scale
                        case_hadrons = pythia.fragment(jet1=jet1, jet2=jet2, scaleIn=scale, weight=chosen_weight)

                        logging.info('Appending event dataframe to hadrons')
                        # Tack case, event, and process details onto the hadron dataframe
                        num_hadrons = len(case_hadrons)
                        event_mult = event_dataframe['mult']
                        event_e2 = event_dataframe['e2']
                        event_psi_e2 = event_dataframe['psi_e2']
                        event_v2 = event_dataframe['v_2']
                        event_psi_2 = event_dataframe['psi_2']
                        event_e3 = event_dataframe['e3']
                        event_psi_e3 = event_dataframe['psi_e3']
                        event_v3 = event_dataframe['v_3']
                        event_psi_3 = event_dataframe['psi_3']
                        event_b = event_dataframe['b']
                        event_ncoll = event_dataframe['ncoll']
                        detail_df = pd.DataFrame(
                            {
                                'hadron_tag': np.random.default_rng().uniform(0, 1000000000000, num_hadrons).astype(int),
                                'drift': np.full(num_hadrons, drift),
                                'el': np.full(num_hadrons, el),
                                'fg': np.full(num_hadrons, fg),
                                'grad': np.full(num_hadrons, grad),
                                'process': np.full(num_hadrons, process_tag),
                                'e_2': np.full(num_hadrons, event_e2),
                                'psi_e2': np.full(num_hadrons, event_psi_e2),
                                'v_2': np.full(num_hadrons, event_v2),
                                'psi_2': np.full(num_hadrons, event_psi_2),
                                'e_3': np.full(num_hadrons, event_e3),
                                'psi_e3': np.full(num_hadrons, event_psi_e3),
                                'v_3': np.full(num_hadrons, event_v3),
                                'psi_3': np.full(num_hadrons, event_psi_3),
                                'mult': np.full(num_hadrons, event_mult),
                                'ncoll': np.full(num_hadrons, event_ncoll),
                                'b': np.full(num_hadrons, event_b),
                                'parent_id': np.empty(num_hadrons),
                                'parent_pt': np.empty(num_hadrons),
                                'parent_pt_f': np.empty(num_hadrons),
                                'parent_phi': np.empty(num_hadrons),
                                'parent_tag': np.empty(num_hadrons),
                                'z': np.empty(num_hadrons)
                            }
                        )
                        case_hadrons = pd.concat([case_hadrons, detail_df], axis=1)

                        logging.info('Hadron z_mean value')
                        # Compute a rough z value for each hadron
                        mean_part_pt = np.mean([jet1.p_T(), jet2.p_T()])
                        case_hadrons['z_mean'] = case_hadrons['pt'] / mean_part_pt

                        logging.info('Hadron phi value')
                        # Compute a phi angle for each hadron
                        case_hadrons['phi_f'] = np.arctan2(case_hadrons['py'].to_numpy().astype(float),
                                                           case_hadrons['px'].to_numpy().astype(float)) + np.pi

                        logging.info('CA-type parent finder')
                        # Apply simplified Cambridge-Aachen-type algorithm to find parent parton
                        for index in case_hadrons.index:
                            min_dR = 10000
                            parent = None

                            # Check the Delta R to each jet
                            # Set parent to the minimum Delta R jet
                            for jet in [jet1, jet2]:
                                jet_rho, jet_phi = jet.polar_mom_coords()
                                dR = delta_R(phi1=case_hadrons.loc[index, 'phi_f'], phi2=jet_phi,
                                             y1=case_hadrons.loc[index, 'y'], y2=0)
                                if dR < min_dR:
                                    min_dR = dR
                                    parent = jet

                            # Save parent info to hadron dataframe
                            case_hadrons.at[index, 'parent_id'] = parent.id
                            case_hadrons.at[index, 'parent_pt'] = parent.p_T0
                            case_hadrons.at[index, 'parent_pt_f'] = parent.p_T()
                            parent_rho, parent_phi = parent.polar_mom_coords()
                            case_hadrons.at[index, 'parent_phi'] = parent_phi
                            case_hadrons.at[index, 'parent_tag'] = parent.tag
                            case_hadrons.at[index, 'z'] = case_hadrons.loc[index, 'pt'] / parent.p_T()  # "Actual" z-value

                    logging.info('Appending case results to process results')
                    if lund_string:
                        process_hadrons = pd.concat([process_hadrons, case_hadrons], axis=0)
                    process_partons = pd.concat([process_partons, case_partons], axis=0)
                    process_run += 1

        except Exception as error:
            logging.info("An error occurred: {}".format(type(error).__name__))  # An error occurred: NameError
            logging.info('- Jet Process Failed -')
            traceback.print_exc()

        if lund_string:
            event_hadrons = pd.concat([event_hadrons, process_hadrons], axis=0)
        event_partons = pd.concat([event_partons, process_partons], axis=0)

        # Declare jet complete
        logging.info('- Jet Process ' + str(process_num) + ' Complete -')

    return event_partons, event_hadrons


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
identifierString = str(int(np.random.uniform(0, 9999999999999)))
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
        logging.info(os.getcwd())

        # Generate a new HIC event and sample config.NUM_SAMPLES jets in it
        # Append returned dataframe to current dataframe
        event_results, event_hadrons = run_event(eventNo=int(identifierString))
        results = pd.concat([results, event_results], axis=0)
        if lund_string:
            hadrons = pd.concat([hadrons, event_hadrons], axis=0)

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
