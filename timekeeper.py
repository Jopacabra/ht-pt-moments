import numpy as np
import pandas as pd
import logging
import plasma_interaction as pi
import config
import xarray as xr
from scipy import interpolate
import os
import traceback


def mean_eloss_rate(pT):
    # Set constants
    mpl = 2.27  # Ballpark value for mean jet path length in QGP

    if pT < 2 or pT > 190:
        return 0.375 * pT / mpl

    else:
        # Get project directory
        project_path = os.path.dirname(os.path.realpath(__file__))

        # Load deltaE / E curve data
        tester_x = np.loadtxt(project_path + '/eoe_data/deltaEoE_thieved_points_PbPb.txt', skiprows=1, usecols=0, delimiter=',')
        tester_y = np.loadtxt(project_path + '/eoe_data/deltaEoE_thieved_points_PbPb.txt', skiprows=1, usecols=1, delimiter=',')

        # Interpolate data
        # This is the delta E / E curve
        interp_func = interpolate.interp1d(x=tester_x, y=tester_y, fill_value="extrapolate")

        # Compute mean energy loss per unit pathlength
        # We take delta E / E, multiply by E, then divide by the mean path length
        mean_eloss_rate_val = interp_func(pT) * pT / mpl

        return mean_eloss_rate_val

def time_loop(event, jet, drift=True, el=True, fg=True, fgqhat=False, scale_drift=1, scale_el=1, el_model='GLV',
              temp_hrg=config.jet.T_HRG, temp_unh=config.jet.T_UNHYDRO):
    #############
    # Time Loop #
    #############
    # Set loop parameters
    tau = config.jet.DTAU  # dt for time loop in fm
    t = event.t0  # Set current time in fm to initial time

    # Initialize counters & values
    t_qgp = -1
    t_hrg = -1
    t_unhydro = -1
    qgp_time_total = 0
    hrg_time_total = 0
    unhydro_time_total = 0
    maxT = 0
    q_el_total = 0
    q_drift_total = 0
    q_drift_abs_total = 0
    q_fg_T_total = 0
    q_fg_T_abs_total = 0
    q_fg_utau_total = 0
    q_fg_utau_abs_total = 0
    q_fg_uperp_total = 0
    q_fg_uperp_abs_total = 0
    q_fgqhat_total = 0
    q_fgqhat_abs_total = 0

    # Initialize flags
    first = True
    qgp_first = True
    hrg_first = True
    unhydro_first = True
    phase = None
    extinguished = False

    # Initialize jet info storage arrays
    time_array = np.array([])
    xpos_array = np.array([])
    ypos_array = np.array([])
    q_drift_array = np.array([])
    q_el_array = np.array([])
    q_fg_T_array = np.array([])
    q_fg_utau_array = np.array([])
    q_fg_uperp_array = np.array([])
    q_fgqhat_array = np.array([])
    pT_array = np.array([])
    temp_seen_array = np.array([])
    grad_perp_T_seen_array = np.array([])
    grad_perp_utau_seen_array = np.array([])
    grad_perp_uperp_seen_array = np.array([])
    u_perp_array = np.array([])
    u_par_array = np.array([])
    u_array = np.array([])
    phase_array = np.array([])

    # Set failsafe values
    rho_final = 0
    phi_final = 0
    pT_final = 0

    # Set mean energy loss rate for Vitev hack
    mean_el_rate = mean_eloss_rate(jet.p_T())

    # Initiate loop
    logging.info('Initiating time loop...')
    while True:
        #logging.info('t = {}'.format(t))
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
        if jet.x > event.xmax or jet.y > event.ymax or jet.x < event.xmin or jet.y < event.ymin:
            logging.info('Jet escaped event space...')
            exit_code = 0
            break
        elif t > event.tf:
            logging.info('Jet escaped event time...')
            if phase == 'qgp':
                exit_code = 3
            else:
                exit_code = 2
            break

        # Record p_T at beginning of step for extinction check
        jet_og_p_T = jet.p_T()

        # For timekeeping in phases, we approximate all time in one step as in one phase
        jet_point = jet.coords3(time=t)
        jet_p_rho, jet_p_phi = jet.polar_mom_coords()
        temp = event.temp(jet_point)
        grad_perp_T = event.grad_perp_T(point=jet_point, phi=jet_p_phi)
        grad_perp_utau = event.grad_perp_u_par(point=jet_point, phi=jet_p_phi)
        grad_perp_uperp = event.grad_perp_u_perp(point=jet_point, phi=jet_p_phi)
        u_perp = event.u_perp(point=jet_point, phi=jet_p_phi)
        u_par = event.u_par(point=jet_point, phi=jet_p_phi)
        u = event.vel(jet_point)

        # Decide phase
        if temp > temp_hrg:
            phase = 'qgp'
        elif temp < temp_hrg and temp > temp_unh:
            phase = 'hrg'
        elif temp < temp_unh and temp > config.transport.hydro.T_SWITCH:
            phase = 'unh'
        else:
            phase = 'vac'

        ############################
        # Perform jet calculations #
        ############################

        if phase == 'qgp':
            # Compute drift, if enabled
            if drift:
                # Compute jet drift integrand in this timestep
                int_drift = pi.jet_drift_integrand(event=event, jet=jet, time=t)

                # Compute Jet drift momentum transferred to jet
                q_drift = float(jet.beta() * tau * int_drift * scale_drift)
            else:
                # Set drift integral and momentum transfer to zero
                int_drift = 0
                q_drift = 0

            # Compute energy loss, if enabled
            if el:
                # Compute energy loss integrand in this timestep
                int_el = pi.energy_loss_integrand(event=event, jet=jet, time=t, tau=tau,
                                                  model=el_model, mean_el_rate=mean_el_rate)

                # Compute energy loss due to gluon exchange with the medium
                q_el = float(jet.beta() * tau * int_el * scale_el)
            else:
                # Set energy loss and el integral to zero
                int_el = 0
                q_el = 0

            if fg:
                # Compute mixed flow-gradient drift integrand in this timestep
                int_fg_T = pi.flowgrad_T_integrand(event=event, jet=jet, time=t)
                int_fg_utau = pi.flowgrad_utau_integrand(event=event, jet=jet, time=t)
                int_fg_uperp = pi.flowgrad_uperp_integrand(event=event, jet=jet, time=t)

                # Compute momentums transferred to jet
                q_fg_T = float(jet.beta() * tau * int_fg_T)
                q_fg_utau = float(jet.beta() * tau * int_fg_utau)
                q_fg_uperp = float(jet.beta() * tau * int_fg_uperp)
            else:
                # Set flow-gradient effects and integral to zero
                int_fg_T = 0
                int_fg_utau = 0
                int_fg_uperp = 0
                q_fg_T = 0
                q_fg_utau = 0
                q_fg_uperp = 0

            if fgqhat:
                # Compute correction to energy loss due to flow-gradient modification
                int_fgqhat = int_el*pi.fg_qhat_mod_factor(event=event, jet=jet, time=t)
                q_fgqhat = float(jet.beta() * tau * int_fgqhat * scale_el)
            else:
                # Set correction to energy loss due to flow-gradient modification to zero
                int_fgqhat = 0
                q_fgqhat = 0

        else:
            # If not in QGP, don't compute any jet-medium interactions
            # If you wanted to add some effects in other phases, they should be computed here
            int_el = 0
            q_el = 0

            int_drift = 0
            q_drift = 0

            int_fg_T = 0
            int_fg_utau = 0
            int_fg_uperp = 0
            q_fg_T = 0
            q_fg_utau = 0
            q_fg_uperp = 0

            int_fgqhat = 0
            q_fgqhat = 0

        ###################
        # Data Accounting #
        ###################
        # Log momentum transfers
        q_el_total += q_el
        q_drift_total += q_drift
        q_drift_abs_total += np.abs(q_drift)
        q_fg_T_total += q_fg_T
        q_fg_T_abs_total += np.abs(q_fg_T)
        q_fg_utau_total += q_fg_utau
        q_fg_utau_abs_total += np.abs(q_fg_utau)
        q_fg_uperp_total += q_fg_uperp
        q_fg_uperp_abs_total += np.abs(q_fg_uperp)
        q_fgqhat_total += q_fgqhat
        q_fgqhat_abs_total += np.abs(q_fgqhat)

        # Check for max temperature
        if temp > maxT:
            maxT = temp[0]

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

        # Record arrays of values from this step for the jet record
        time_array = np.append(time_array, t)
        xpos_array = np.append(xpos_array, jet.x)
        ypos_array = np.append(ypos_array, jet.y)
        q_drift_array = np.append(q_drift_array, q_drift)
        q_el_array = np.append(q_el_array, q_el)
        q_fg_T_array = np.append(q_fg_T_array, q_fg_T)
        q_fg_utau_array = np.append(q_fg_utau_array, q_fg_utau)
        q_fg_uperp_array = np.append(q_fg_uperp_array, q_fg_uperp)
        q_fgqhat_array = np.append(q_fgqhat_array, q_fgqhat)
        pT_array = np.append(pT_array, jet.p_T())
        temp_seen_array = np.append(temp_seen_array, temp)
        grad_perp_T_seen_array = np.append(grad_perp_T_seen_array, grad_perp_T)
        grad_perp_utau_seen_array = np.append(grad_perp_utau_seen_array, grad_perp_utau)
        grad_perp_uperp_seen_array = np.append(grad_perp_uperp_seen_array, grad_perp_uperp)
        u_perp_array = np.append(u_perp_array, u_perp)
        u_par_array = np.append(u_par_array, u_par)
        u_array = np.append(u_array, u)
        phase_array = np.append(phase_array, phase)

        #########################
        # Change Jet Parameters #
        #########################
        # Propagate jet position
        jet.prop(tau=tau)

        # Change jet momentum to reflect energy loss
        jet.add_q_par(q_par=q_el)
        jet.add_q_par(q_par=q_fgqhat)

        # Change jet momentum to reflect drift effects
        # If not computed, q values go to zero.
        jet.add_q_perp(q_perp=q_drift)
        jet.add_q_perp(q_perp=q_fg_T)
        jet.add_q_perp(q_perp=q_fg_utau)
        jet.add_q_perp(q_perp=q_fg_uperp)

        # Check if the jet would be extinguished (prevents flipping directions
        # when T >> p_T, since q_el has no p_T dependence):
        # If the jet lost more energy this step than it had
        # at the beginning of the step, we extinguish the jet and end things
        if np.abs(q_el) >= jet_og_p_T:
            logging.info('Jet extinguished')
            jet.p_x = 0
            jet.p_y = 0
            extinguished = True
            exit_code = 1
            break

        ###############
        # Timekeeping #
        ###############
        t += tau

        # Get final jet parameters
        rho_final, phi_final = jet.polar_mom_coords()
        pT_final = jet.p_T()

    logging.info('Time loop complete...')

    # Create momentPlasma results dataframe
    try:
        print('Making dataframe...')
        jet_dataframe = pd.DataFrame(
            {
                "jetNo": [int(jet.no)],
                "tag": [int(jet.tag)],
                "weight": [float(jet.weight)],
                "id": [int(jet.id)],
                "pt_0": [float(jet.p_T0)],
                "pt_f": [float(pT_final)],
                "q_el": [float(q_el_total)],
                "q_drift": [float(q_drift_total)],
                "q_drift_abs": [float(q_drift_abs_total)],
                "q_fg_T": [float(q_fg_T_total)],
                "q_fg_T_abs": [float(q_fg_T_abs_total)],
                "q_fg_utau": [float(q_fg_utau_total)],
                "q_fg_utau_abs": [float(q_fg_utau_abs_total)],
                "q_fg_uperp": [float(q_fg_uperp_total)],
                "q_fg_uperp_abs": [float(q_fg_uperp_abs_total)],
                "q_fgqhat": [float(q_fgqhat_total)],
                "q_fgqhat_abs": [float(q_fgqhat_abs_total)],
                "extinguished": [bool(extinguished)],
                "x_0": [float(jet.x_0)],
                "y_0": [float(jet.y_0)],
                "phi_0": [float(jet.phi_0)],
                "phi_f": [float(phi_final)],
                "t_qgp": [float(t_qgp)],
                "t_hrg": [float(t_hrg)],
                "t_unhydro": [float(t_unhydro)],
                "time_total_plasma": [float(qgp_time_total)],
                "time_total_hrg": [float(hrg_time_total)],
                "time_total_unhydro": [float(unhydro_time_total)],
                "Tmax_jet": [float(maxT)],
                "initial_time": [float(event.t0)],
                "final_time": [float(event.tf)],
                "tau": [float(config.jet.DTAU)],
                "Tmax_event": [float(event.max_temp())],
                "drift": [bool(drift)],
                "el": [bool(el)],
                "fg": [bool(fg)],
                "fgqhat": [bool(fgqhat)],
                "exit": [int(exit_code)],
                "g": [float(config.constants.G)]
            }
        )

        logging.info('Pandas dataframe generated...')

    except Exception as error:
        logging.info("An error occurred: {}".format(type(error).__name__))  # An error occurred: NameError
        logging.info('- Jet Dataframe Creation Failed -')
        traceback.print_exc()

    # Create and store jet record xarray
    # define data with variable attributes
    logging.info('Creating xarray jet record...')
    data_vars = {'x': (['time'], xpos_array,
                       {'units': 'fm',
                        'long_name': 'x position coordinate'}),
                 'y': (['time'], ypos_array,
                       {'units': 'fm',
                        'long_name': 'y position coordinate'}),
                 'q_drift': (['time'], q_drift_array,
                             {'units': 'GeV',
                              'long_name': 'Momentum obtained by the jet at this timestep due to flow drift'}),
                 'q_fg_T': (['time'], q_fg_T_array,
                          {'units': 'GeV',
                           'long_name': 'Momentum obtained by the jet at this timestep due to flow-grad_T drift'}),
                 'q_fg_utau': (['time'], q_fg_utau_array,
                          {'units': 'GeV',
                           'long_name': 'Momentum obtained by the jet at this timestep due to flow-grad_utau drift'}),
                 'q_fg_uperp': (['time'], q_fg_uperp_array,
                          {'units': 'GeV',
                           'long_name': 'Momentum obtained by the jet at this timestep due to flow-grad_uperp drift'}),
                 'q_el': (['time'], q_el_array,
                            {'units': 'GeV',
                             'long_name': 'Momentum obtained by the jet at this timestep due to energy loss'}),
                 'q_fgqhat': (['time'], q_fgqhat_array,
                          {'units': 'GeV',
                           'long_name': 'Momentum obtained by the jet at this timestep due to fg mod to energy loss'}),
                 'pT': (['time'], pT_array,
                          {'units': 'GeV',
                           'long_name': 'Transverse momentum of the jet at this timestep'}),
                 'temp': (['time'], temp_seen_array,
                          {'units': 'GeV',
                           'long_name': 'Temperature seen by the jet at this timestep'}),
                 'grad_perp_temp': (['time'], grad_perp_T_seen_array,
                          {'units': 'GeV/fm',
                           'long_name': 'Gradient of Temperature perp. to jet seen by the jet at this timestep'}),
                 'grad_perp_utau': (['time'], grad_perp_utau_seen_array,
                                    {'units': 'GeV/fm',
                                     'long_name': 'Gradient of utau perp. to jet seen by the jet at this timestep'}),
                 'grad_perp_uperp': (['time'], grad_perp_uperp_seen_array,
                                    {'units': 'GeV/fm',
                                     'long_name': 'Gradient of uperp perp. to jet seen by the jet at this timestep'}),
                 'u_perp': (['time'], u_perp_array,
                            {'units': 'GeV',
                             'long_name': 'Temperature seen by the jet at this timestep'}),
                 'u_par': (['time'], u_par_array,
                           {'units': 'GeV',
                            'long_name': 'Temperature seen by the jet at this timestep'}),
                 'u': (['time'], u_array,
                       {'units': 'GeV',
                        'long_name': 'Temperature seen by the jet at this timestep'}),
                 'phase': (['time'], phase_array,
                           {'units': 'qgp = Quark Gluon Plasma, hrg = HadRon Gas, unh = UNHydrodynamic hadron gas, vac = below unh cutoff / vacuum',
                            'long_name': 'Phase seen by the jet at this timestep'})
                 }

    # define coordinates
    coords = {'time': (['time'], time_array)}

    # define global attributes
    attrs = {'property_name': 'value'}

    # create dataset
    jet_xarray = xr.Dataset(data_vars=data_vars,
                            coords=coords,
                            attrs=attrs)

    jet.record = jet_xarray

    logging.info('Xarray dataframe generated...')

    return jet_dataframe, jet_xarray
