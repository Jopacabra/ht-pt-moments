import numpy as np
import pandas as pd
import logging
import jets
import pythia


# Computes angular distance-ish quantity
def delta_R(phi1, phi2, y1, y2):
    dphi = np.abs(phi1 - phi2)
    if (dphi > np.pi):
        dphi = 2*np.pi - dphi
    drap = y1 - y2
    return np.sqrt(dphi * dphi + drap * drap)

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

################
# Jet Analysis #
################
# Oversample the background with jets

process_num = 1000000
for jetNo in range(0, process_num):
    # Create unique jet tag
    process_tag = int(np.random.uniform(0, 1000000000000))
    logging.info('- Jet Process {} Start -'.format(jetNo))

    try:
        #########################
        # Create new scattering #
        #########################
        particles, weight = pythia.scattering()
        particle_tags = np.random.default_rng().uniform(0, 1000000000000, len(particles)).astype(int)
        process_hadrons = pd.DataFrame({})



        i = 0
        for index, particle in particles.iterrows():
            # Only do the things for the particle output
            particle_status = particle['status']
            particle_tag = int(particle_tags[i])
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
            x0 = 0
            y0 = 0

            # Read jet production angle
            phi_0 = np.arctan2(particle['py'], particle['px']) + np.pi

            # Yell about your selected jet
            logging.info('Pilot parton: {}, pT: {} GeV'.format(chosen_pilot, chosen_e))

            el_model = 'SGLV'

            # Log jet number and case description

            # Create the jet object
            jet = jets.jet(x_0=x0, y_0=y0, phi_0=phi_0, p_T0=chosen_e, tag=particle_tag, no=jetNo, part=chosen_pilot,
                           weight=chosen_weight)

            # Save jet pair
            if i == 4:
                jet1 = jet
            elif i == 5:
                jet2 = jet

            i += 1

        logging.info('Hadronizing...')
        # Hadronize jet pair
        scale = particles['scaleIn'].to_numpy()[-1]  # use last particle to set hard process scale
        case_hadrons = pythia.fragment(jet1=jet1, jet2=jet2, scaleIn=scale, weight=chosen_weight)

        logging.info('Appending event dataframe to hadrons')
        # Tack case, event, and process details onto the hadron dataframe
        num_hadrons = len(case_hadrons)
        detail_df = pd.DataFrame(
            {
                'hadron_tag': np.random.default_rng().uniform(0, 1000000000000, num_hadrons).astype(int),
                'process': np.full(num_hadrons, process_tag),
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
        process_hadrons = pd.concat([process_hadrons, case_hadrons], axis=0)

    except Exception as error:
        logging.info("An error occurred: {}".format(type(error).__name__))  # An error occurred: NameError
        logging.info('- Jet Process Failed -')

    event_hadrons = pd.concat([event_hadrons, process_hadrons], axis=0)

    # Declare jet complete
    logging.info('- Jet Process ' + str(jetNo) + ' Complete -')


# Save the dataframe into the identified results folder
logging.info('Saving progress...')
hadrons_df = event_hadrons
logging.debug(hadrons_df)
logging.info('Converting datatypes to reasonable formats...')
hadrons_df = hadrons_df.convert_dtypes()
logging.info('Downcasting numeric values to minimum memory size...')
hadrons_df = downcast_numerics(hadrons_df)
logging.info('Writing pickles...')
hadrons_df.to_parquet('pp_hadrons.parquet')
