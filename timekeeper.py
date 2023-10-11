import numpy as np
import pandas as pd
import logging
import plasma_interaction as pi
import config
import xarray as xr
from scipy import interpolate
import os


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

def time_loop(event, jet, drift=True, grad=False, el=True, fg=True, scale_drift=1, scale_grad=1, scale_el=1, el_model='SGLV',
              temp_hrg=config.transport.hydro.T_HRG, temp_unh=config.transport.hydro.T_UNHYDRO):
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
    q_el_total = 0
    q_drift_total = 0
    q_drift_abs_total = 0
    q_grad_total = 0
    q_grad_abs_total = 0
    q_fg_total = 0
    q_fg_abs_total = 0

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
    q_grad_array = np.array([])
    q_el_array = np.array([])
    q_fg_array = np.array([])
    int_drift_array = np.array([])
    int_grad_array = np.array([])
    int_el_array = np.array([])
    pT_array = np.array([])
    temp_seen_array = np.array([])
    grad_perp_T_seen_array = np.array([])
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
        u_perp = event.u_perp(point=jet_point, phi=jet_p_phi)
        u_par = event.u_par(point=jet_point, phi=jet_p_phi)
        u = event.vel(jet_point)

        # Decide phase
        if temp > temp_hrg:
            phase = 'qgp'
        elif temp < temp_hrg and temp > temp_unh:
            phase = 'hrg'
        elif temp < temp_unh and temp > config.transport.hydro.T_END:
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

            # Compute gradient drift, if enabled
            if grad:
                # Compute gradient drift integrand in this timestep
                int_grad = pi.grad_integrand(event=event, jet=jet, time=t, tau=tau)

                # Compute Jet gradient drift momentum transferred to jet
                q_grad = float(jet.beta() * tau * int_grad * scale_grad)
            else:
                # Set gradient drift integral and momentum transfer to zero
                int_grad = 0
                q_grad = 0

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
                int_fg = pi.flowgrad_drift_integrand(event=event, jet=jet, time=t, tau=tau)

                # Compute momentum transferred to jet
                q_fg = float(jet.beta() * tau * int_fg)
            else:
                # Set flow-gradient effects and integral to zero
                int_fg = 0
                q_fg = 0

        else:
            # If not in QGP, don't compute any jet-medium interactions
            # If you wanted to add some effects in other phases, they should be computed here
            int_drift = 0
            int_grad = 0
            int_el = 0
            int_fg = 0
            q_drift = 0
            q_grad = 0
            q_el = 0
            q_fg = 0

        ###################
        # Data Accounting #
        ###################
        # Log momentum transfers
        q_el_total += q_el
        q_drift_total += q_drift
        q_drift_abs_total += np.abs(q_drift)
        q_grad_total += q_grad
        q_grad_abs_total += np.abs(q_grad)
        q_fg_total += q_fg
        q_fg_abs_total += np.abs(q_fg)

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
        q_grad_array = np.append(q_grad_array, q_grad)
        q_el_array = np.append(q_el_array, q_el)
        q_fg_array = np.append(q_fg_array, q_fg)
        int_drift_array = np.append(int_drift_array, int_drift)
        int_grad_array = np.append(int_grad_array, int_drift)
        int_el_array = np.append(int_el_array, int_el)
        pT_array = np.append(pT_array, jet.p_T())
        temp_seen_array = np.append(temp_seen_array, temp)
        grad_perp_T_seen_array = np.append(grad_perp_T_seen_array, grad_perp_T)
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

        # Change jet momentum to reflect drift effects
        # If not computed, q values go to zero.
        jet.add_q_perp(q_perp=q_drift)
        jet.add_q_perp(q_perp=q_grad)
        jet.add_q_perp(q_perp=q_fg)

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
    jet_dataframe = pd.DataFrame(
        {
            "eventNo": [event.name],
            "jetNo": [jet.no],
            "jet_tag": [jet.tag],
            "jet_weight": [jet.weight],
            "jet_particle": [jet.part],
            "jet_mass": [jet.m],
            "jet_pT": [jet.p_T0],
            "jet_pT_f": [pT_final],
            "q_el": [float(q_el_total)],
            "q_drift": [float(q_drift_total)],
            "q_drift_abs": [float(q_drift_abs_total)],
            "q_grad": [float(q_grad_total)],
            "q_grad_abs": [float(q_grad_abs_total)],
            "q_fg": [float(q_fg_total)],
            "q_fg_abs": [float(q_fg_abs_total)],
            "extinguished": [extinguished],
            "X0": [jet.x_0],
            "Y0": [jet.y_0],
            "phi_0": [jet.phi_0],
            "phi_f": [phi_final],
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
            "Tmax_event": [event.max_temp()],
            "drift": [drift],
            "grad": [grad],
            "el": [el],
            "fg": [fg],
            "el_model": [el_model],
            "k_drift": [scale_drift*config.constants.K_DRIFT],
            "k_BBMG": [scale_el * config.constants.K_BBMG],
            "exit_code": [exit_code]
        }
    )

    logging.info('Pandas dataframe generated...')

    # Create and store jet record xarray
    # define data with variable attributes
    data_vars = {'x': (['time'], xpos_array,
                       {'units': 'fm',
                        'long_name': 'x position coordinate'}),
                 'y': (['time'], ypos_array,
                       {'units': 'fm',
                        'long_name': 'y position coordinate'}),
                 'q_drift': (['time'], q_drift_array,
                             {'units': 'GeV',
                              'long_name': 'Momentum obtained by the jet at this timestep due to flow drift'}),
                 'q_grad': (['time'], q_grad_array,
                             {'units': 'GeV',
                              'long_name': 'Momentum obtained by the jet at this timestep due to gradient drift'}),
                 'q_fg': (['time'], q_fg_array,
                          {'units': 'GeV',
                           'long_name': 'Momentum obtained by the jet at this timestep due to flow-gradient drift'}),
                 'q_EL': (['time'], q_el_array,
                            {'units': 'GeV',
                             'long_name': 'Momentum obtained by the jet at this timestep due to BBMG energy loss'}),
                 'int_drift': (['time'], int_drift_array,
                             {'units': 'GeV/fm',
                              'long_name': 'Drift integrand seen by the jet at this timestep due to jet drift'}),
                 'int_grad': (['time'], int_grad_array,
                               {'units': 'GeV/fm',
                                'long_name': 'Gradient drift integrand seen by the jet at this timestep due to jet drift'}),
                 'int_EL': (['time'], int_el_array,
                            {'units': 'GeV',
                             'long_name': 'BBMG integrand seen by the jet at this timestep due to BBMG energy loss'}),
                 'pT': (['time'], pT_array,
                          {'units': 'GeV',
                           'long_name': 'Transverse momentum of the jet at this timestep'}),
                 'temp': (['time'], temp_seen_array,
                          {'units': 'GeV',
                           'long_name': 'Temperature seen by the jet at this timestep'}),
                 'grad_perp_temp': (['time'], grad_perp_T_seen_array,
                          {'units': 'GeV/fm',
                           'long_name': 'Gradient of Temperature perp. to jet seen by the jet at this timestep'}),
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
