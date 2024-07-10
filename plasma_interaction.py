import os
import numpy as np
import config
import utilities
import logging
from scipy.interpolate import RegularGridInterpolator

# Function to return DeBye mass at a particular point
# Ref - https://inspirehep.net/literature/1725162
def mu_DeBye(T, g=None):
    Nf = 2  # Number of light quark flavors
    if g is None:
        g = config.constants.G
    debye_mass = g * T * np.sqrt(1 + Nf /6)
    return debye_mass

# Function to return total cross section at a particular point for parton and *gluon* in medium
# Total GW cross section, as per Sievert, Yoon, et. al.
# Specify med_parton either 'g' for medium gluon or 'q' for generic light (?) quark in medium
# https://inspirehep.net/literature/1725162
def sigma(temp, parton_type, med_parton='g'):
    """
    We select the appropriate cross-section for a known parton and
    known medium parton specified when called
    """
    coupling = config.constants.G

    sigma_gg_gg = (9/(32 * np.pi)) * coupling ** 4 / (mu_DeBye(T=temp) ** 2)
    sigma_qg_qg = (1/(8 * np.pi)) * coupling ** 4 / (mu_DeBye(T=temp) ** 2)
    sigma_qq_qq = (1/(18 * np.pi)) * coupling ** 4 / (mu_DeBye(T=temp) ** 2)

    if parton_type == 'g' and med_parton == 'g':
        # gg -> gg cross-section
        cross_section = sigma_gg_gg
    elif parton_type == 'q' and med_parton == 'g':
        # qg -> qg cross-section
        cross_section = sigma_qg_qg
    elif parton_type == 'g' and med_parton == 'q':
        # qg -> qg cross-section
        cross_section = sigma_qg_qg
    elif parton_type == 'q' and med_parton == 'q':
        # qq -> qq cross-section
        cross_section = sigma_qq_qq
    else:
        logging.debug('Unknown parton scattering cs... Using gg->gg scattering cross section')
        cross_section = sigma_gg_gg

    return cross_section

# Function to return partial density at a particular point for given medium partons
# Chosen to be ideal gluon gas dens. as per Sievert, Yoon, et. al.
def rho(temp, med_parton='g'):
    if med_parton == 'g':
        density = 1.202056903159594 * 16 * (1 / (np.pi ** 2)) * temp ** 3
    elif med_parton == 'q':
        density = 1.202056903159594 * (3/4) * 24 * (1 / (np.pi ** 2)) * temp ** 3
    else:
        # Return 0
        density = 0
    return density

# Function to return inverse QGP drift mean free path in units of GeV^{-1}
# Total GW cross section, as per Sievert, Yoon, et. al.
def inv_lambda(event=None, parton_type=None, point=None, med_parton='all', T=None):
    """
    We apply a reciprocal summation between the cross-section times density for a medium gluon and for a medium quark
    to get the mean free path as in https://inspirehep.net/literature/1725162
    """
    if T is not None and event is None:
        temp = T
    else:
        temp = event.temp(point)

    if med_parton == 'all':
        return (sigma(temp, parton_type, med_parton='g') * rho(temp, med_parton='g')
                + sigma(temp, parton_type, med_parton='q') * rho(temp, med_parton='q'))
    else:
        return sigma(temp, parton_type, med_parton=med_parton) * rho(temp, med_parton=med_parton)

# Define integrand for mean q_drift (k=0 moment)
def drift_integrand(event, parton, time):
    FmGeV = 1/0.19732687

    # Get parton coordinates
    point = parton.coords3(time=time)
    p_rho, p_phi = parton.polar_mom_coords()
    E = parton.p_T()
    beta = parton.beta()

    # Average medium parameters
    u_perp = utilities.dtau_avg(func=lambda x : event.u_perp(point=x, phi=p_phi), point=point, phi=p_phi,
                                dtau=config.jet.DTAU, beta=beta)
    u_tau = utilities.dtau_avg(func=lambda x : event.u_par(point=x, phi=p_phi), point=point, phi=p_phi,
                                dtau=config.jet.DTAU, beta=beta)
    mu = utilities.dtau_avg(func=lambda x : mu_DeBye(T=event.temp(x)), point=point, phi=p_phi,
                            dtau=config.jet.DTAU, beta=beta)
    inv_lambda_val = utilities.dtau_avg(func=lambda x : inv_lambda(event=event, parton_type=parton.part, point=x),
                                        point=point, phi=p_phi, dtau=config.jet.DTAU, beta=beta)

    # Source link? -- Converts factor of fermi from integral to factor of GeV^{-1}
    return ((FmGeV) * (1 / parton.p_T()) * config.jet.K_F_DRIFT
            * (3 * np.log(E/mu)
               * (u_perp / (1 - u_tau))
               * (mu**2)
               * inv_lambda_val))

# Define integrand for mean flow-grad_uT drift
def flowgrad_T_integrand(event, parton, time):
    FmGeV = 1/0.19732687

    # Get parton coordinates
    point = parton.coords3(time=time)
    p_rho, p_phi = parton.polar_mom_coords()
    E = parton.p_T()
    beta = parton.beta()

    # Average medium parameters
    T = utilities.dtau_avg(func=event.temp, point=point, phi=p_phi, dtau=config.jet.DTAU, beta=beta)
    u_perp = utilities.dtau_avg(func=lambda x: event.u_perp(point=x, phi=p_phi), point=point, phi=p_phi,
                                dtau=config.jet.DTAU, beta=beta)
    u_tau = utilities.dtau_avg(func=lambda x: event.u_par(point=x, phi=p_phi), point=point, phi=p_phi,
                               dtau=config.jet.DTAU, beta=beta)
    mu = utilities.dtau_avg(func=lambda x: mu_DeBye(T=event.temp(x)), point=point, phi=p_phi,
                            dtau=config.jet.DTAU, beta=beta)
    inv_lambda_val = utilities.dtau_avg(func=lambda x: inv_lambda(event=event, parton_type=parton.part, point=x),
                                        point=point, phi=p_phi, dtau=config.jet.DTAU, beta=beta)
    grad_perp_temp = utilities.dtau_avg(func=lambda x: event.grad_perp_T(point=x, phi=p_phi),
                                        point=point, phi=p_phi, dtau=config.jet.DTAU, beta=beta)

    return - ((FmGeV) * (3 / E) * config.jet.K_FG_DRIFT * (time - event.t0)
              * 3 * grad_perp_temp * ((u_perp**2)/((1 - u_tau)**2)) * (1/T)
              * (mu**2) * inv_lambda_val
              * np.log(E / mu))

# Define integrand for mean flow-grad_utau drift
def flowgrad_utau_integrand(event, parton, time):
    FmGeV = 1/0.19732687

    # Get parton coordinates
    point = parton.coords3(time=time)
    p_rho, p_phi = parton.polar_mom_coords()
    E = parton.p_T()
    beta = parton.beta()

    # Average medium parameters
    #T = utilities.dtau_avg(func=event.temp, point=point, phi=p_phi, dtau=config.jet.DTAU, beta=beta)
    u_perp = utilities.dtau_avg(func=lambda x: event.u_perp(point=x, phi=p_phi), point=point, phi=p_phi,
                                dtau=config.jet.DTAU, beta=beta)
    u_tau = utilities.dtau_avg(func=lambda x: event.u_par(point=x, phi=p_phi), point=point, phi=p_phi,
                               dtau=config.jet.DTAU, beta=beta)
    mu = utilities.dtau_avg(func=lambda x: mu_DeBye(T=event.temp(x)), point=point, phi=p_phi,
                            dtau=config.jet.DTAU, beta=beta)
    inv_lambda_val = utilities.dtau_avg(func=lambda x: inv_lambda(event=event, parton_type=parton.part, point=x),
                                        point=point, phi=p_phi, dtau=config.jet.DTAU, beta=beta)
    grad_perp_u_tau = utilities.dtau_avg(func=lambda x: event.grad_perp_u_par(point=x, phi=p_phi),
                                        point=point, phi=p_phi, dtau=config.jet.DTAU, beta=beta)

    # Source link? -- Converts factor of fermi from integral to factor of GeV^{-1}
    return - ((FmGeV) * (3 / E) * config.jet.K_FG_DRIFT * (time - event.t0)
              * 2 * grad_perp_u_tau * ((u_perp**2)/((1 - u_tau)**3))
              * (mu**2) * inv_lambda_val
              * np.log(E / mu))

# Define integrand for mean flow-grad_uperp drift
def flowgrad_uperp_integrand(event, parton, time):
    FmGeV = 1/0.19732687

    # Get parton coordinates
    point = parton.coords3(time=time)
    p_rho, p_phi = parton.polar_mom_coords()
    E = parton.p_T()
    beta = parton.beta()

    # Average medium parameters
    # T = utilities.dtau_avg(func=event.temp, point=point, phi=p_phi, dtau=config.jet.DTAU, beta=beta)
    u_perp = utilities.dtau_avg(func=lambda x: event.u_perp(point=x, phi=p_phi), point=point, phi=p_phi,
                                dtau=config.jet.DTAU, beta=beta)
    u_tau = utilities.dtau_avg(func=lambda x: event.u_par(point=x, phi=p_phi), point=point, phi=p_phi,
                               dtau=config.jet.DTAU, beta=beta)
    mu = utilities.dtau_avg(func=lambda x: mu_DeBye(T=event.temp(x)), point=point, phi=p_phi,
                            dtau=config.jet.DTAU, beta=beta)
    inv_lambda_val = utilities.dtau_avg(func=lambda x: inv_lambda(event=event, parton_type=parton.part, point=x),
                                        point=point, phi=p_phi, dtau=config.jet.DTAU, beta=beta)
    grad_perp_u_perp = utilities.dtau_avg(func=lambda x: event.grad_perp_u_perp(point=x, phi=p_phi),
                                          point=point, phi=p_phi, dtau=config.jet.DTAU, beta=beta)

    # Source link? -- Converts factor of fermi from integral to factor of GeV^{-1}
    return - ((FmGeV) * (3 / E) * config.jet.K_FG_DRIFT * (time - event.t0)
              * 2 * grad_perp_u_perp * (u_perp/((1 - u_tau)**2))
              * (mu**2) * inv_lambda_val
              * np.log(E / mu))

# Function to sample ebe fluctuation zeta parameter for energy loss integral
def zeta(q=0, maxAttempts=5, batch=1000):
    # Special cases making things easier
    if q == 0:
        rng = np.random.default_rng()
        return rng.random() * 2
    elif q == -1:
        return 1

    attempt = 0
    while attempt < maxAttempts:
        # Generate random point in 3D box of l = w = gridWidth and height maximum temp.^6
        # Origin at center of bottom of box
        pointArray = utilities.random_2d(num=batch, boxSize=q + 2, maxProb=1)
        for point in pointArray:
            x = point[0]
            y = point[1]
            targetVal = ((1 + q) / ((q + 2) ** (1 + q))) * ((q + 2 - x) ** q)

            # Check if point under 2D temp PDF curve
            if float(y) < float(targetVal):
                # If under curve, accept point and return
                # print("Attempt " + str(attempt) + " successful with point " + str(i) + "!!!")
                # print(point)
                # print("Random height: " + str(zPoints[i]))
                # print("Target <= height: " + str(float(targetTemp)))
                return x
        print("Zeta Parameter Sample Attempt: " + str(attempt) + " failed.")
        attempt += 1
    print("Catastrophic error in zeta parameter sampling!")
    print("AHHHHHHHHHHHHHHH!!!!!!!!!!!")
    return 0


# Integrand for energy loss
def energy_loss_integrand(event, parton, time, tau, model='BBMG', fgqhat=False, mean_el_rate=0):
    FmGeV = 1/0.19732687

    # Get parton coordinates
    point = parton.coords3(time=time)
    p_rho, p_phi = parton.polar_mom_coords()
    E = parton.p_T()
    beta = parton.beta()

    # Average medium parameters
    T = utilities.dtau_avg(func=event.temp, point=point, phi=p_phi, dtau=config.jet.DTAU, beta=beta)
    mu = utilities.dtau_avg(func=lambda x: mu_DeBye(T=event.temp(x)), point=point, phi=p_phi,
                            dtau=config.jet.DTAU, beta=beta)
    inv_lambda_val = utilities.dtau_avg(func=lambda x: inv_lambda(event=event, parton_type=parton.part, point=x),
                                        point=point, phi=p_phi, dtau=config.jet.DTAU, beta=beta)
    vel = utilities.dtau_avg(func=event.vel, point=point, phi=p_phi, dtau=config.jet.DTAU, beta=beta)

    # Select energy loss model and return appropriate energy loss
    if model == 'BBMG':
        # Note that we apply FERMI GeV twice... Once for the t factor, once for the (int dt).
        return (config.jet.K_BBMG * (-1) * ((FmGeV) ** 2) * time * (T ** 3)
                * zeta(q=-1) * (1 / np.sqrt(1 - (vel**2)))
                * (1))
    elif model == 'GLV':
        # https://inspirehep.net/literature/539404
        # Note that we apply FERMItoGeV twice... Once for the t factor, once for the (int dt).
        # Set C_R, "quadratic Casimir of the representation R of SU(3) for the parton"
        if parton.part == 'g':
            # For a gluon it's the adjoint representation C_A = N_c = 3
            CR = 3
        else:
            # For a quark it's the fundamental representation C_F = 4/3 in QCD
            CR = 4/3

        # Set alpha_s
        alphas = (config.constants.G**2) / (4*np.pi)

        # Calculate and return energy loss per unit length of this step.
        return (-1)*(CR * alphas / 2) * (((1 / FmGeV) ** 2)
                                         * (time - event.t0)
                                         * (mu**2)
                                         * inv_lambda_val
                                         * np.log(E / mu))
    else:
        return 0

# # Integrand for gradient deflection to 2nd order in opacity
# # Note - first moment is zero. Essentially computing cuberoot(q_{grad}^3) as scale approx.
# def grad_integrand(event, parton, time, tau):
#     point = parton.coords3(time=time)
#     p_rho, p_phi = parton.polar_mom_coords()
#     FmGeV = 0.19732687
#
#     '''
#     Omega here is the characteristic width of the gaussian approximating the jet width spectrum.
#     For a simple first investigation of the order of magnitude of the gradient effects, we
#     assume that this is equivalent to the gluon saturation scale in a p-X collision system.
#
#     We take the gluon saturation scale from the pocket equation in Eq. 3.6 here:
#     https://inspirehep.net/literature/1206324
#     fit by eye to the results in Fig. 3.9 left for Au at x = 0.0001, which roughly equates to
#     the region of 1 GeV jets in Au Au collisions at sqrt(s) == 5.02 TeV
#     '''
#
#     # Select proper saturation scale from scaled pocket equation
#     if config.transport.trento.PROJ1 == 'Pb' and config.transport.trento.PROJ2 == 'Pb':
#         A = 208
#     else:
#         A = 197
#     x = parton.p_T() / config.constants.ROOT_S
#     omega = 0.01675 * ((A / x) ** (1 / 3))
#
#     first_order_q = FmGeV*(((2 * (omega**2) * tau * (event.mu(point=point)**2)
#                              * event.grad_perp_rho(point=point, phi=p_phi, med_parton='q')
#                              * inv_lambda(event=event, parton=parton, point=point, med_parton='q'))
#                             / (parton.p_T() * event.rho(point, med_parton='q')))
#                            * np.log(parton.p_T() / event.mu(point=point)))
#
#     first_order_g = FmGeV*(((2 * (omega**2) * tau * (event.mu(point=point)**2)
#                              * event.grad_perp_rho(point=point, phi=p_phi, med_parton='g')
#                              * inv_lambda(event=event, parton=parton, point=point, med_parton='g'))
#                             / (parton.p_T() * event.rho(point, med_parton='g')))
#                            * np.log(parton.p_T() / event.mu(point=point)))
#
#     second_order_q = (FmGeV**2) * ((tau**2) * (event.mu(point=point)**4) * event.grad_perp_rho(point=point, phi=p_phi, med_parton='q')
#                                    * (inv_lambda(event=event, parton=parton, point=point, med_parton='q') ** 2)
#                                    * (np.log(parton.p_T() / event.mu(point=point)) ** 2)
#                                    / (2 * parton.p_T() * (event.rho(point, med_parton='q'))))
#
#     second_order_g = (FmGeV**2) * ((tau ** 2) * (event.mu(point=point) ** 4) * event.grad_perp_rho(point=point, phi=p_phi, med_parton='g')
#                                    * (inv_lambda(event=event, parton=parton, point=point, med_parton='g') ** 2)
#                                    * (np.log(parton.p_T() / event.mu(point=point)) ** 2)
#                                    / (2 * parton.p_T() * (event.rho(point, med_parton='g'))))
#
#     return np.cbrt(first_order_q + first_order_g + second_order_q + second_order_g)

# Modification factor for energy loss due to gradients of temperature
def fg_T_qhat_mod_factor(event, parton, time):

    # Get parton coordinates
    point = parton.coords3(time=time)
    p_rho, p_phi = parton.polar_mom_coords()
    beta = parton.beta()

    # Average medium parameters
    T = utilities.dtau_avg(func=event.temp, point=point, phi=p_phi, dtau=config.jet.DTAU, beta=beta)
    u_perp = utilities.dtau_avg(func=lambda x: event.u_perp(point=x, phi=p_phi), point=point, phi=p_phi,
                                dtau=config.jet.DTAU, beta=beta)
    u_tau = utilities.dtau_avg(func=lambda x: event.u_par(point=x, phi=p_phi), point=point, phi=p_phi,
                               dtau=config.jet.DTAU, beta=beta)
    grad_perp_temp = utilities.dtau_avg(func=lambda x: event.grad_perp_T(point=x, phi=p_phi),
                                          point=point, phi=p_phi, dtau=config.jet.DTAU, beta=beta)

    return (-1) * (time - event.t0) * (3 * grad_perp_temp * (u_perp / (1-u_tau)) * (1/T))


# Modification factor for energy loss due to gradients of utau
def fg_utau_qhat_mod_factor(event, parton, time):
    # Get parton coordinates
    point = parton.coords3(time=time)
    p_rho, p_phi = parton.polar_mom_coords()
    beta = parton.beta()

    # Average medium parameters
    u_perp = utilities.dtau_avg(func=lambda x: event.u_perp(point=x, phi=p_phi), point=point, phi=p_phi,
                                dtau=config.jet.DTAU, beta=beta)
    u_tau = utilities.dtau_avg(func=lambda x: event.u_par(point=x, phi=p_phi), point=point, phi=p_phi,
                               dtau=config.jet.DTAU, beta=beta)
    grad_perp_u_tau = utilities.dtau_avg(func=lambda x: event.grad_perp_u_par(point=x, phi=p_phi),
                                         point=point, phi=p_phi, dtau=config.jet.DTAU, beta=beta)

    return (-1) * (time - event.t0) * (grad_perp_u_tau * (u_perp / ((1-u_tau)**2)))

# Modification factor for energy loss due to gradients of uperp
def fg_uperp_qhat_mod_factor(event, parton, time):
    # Get parton coordinates
    point = parton.coords3(time=time)
    p_rho, p_phi = parton.polar_mom_coords()
    beta = parton.beta()

    # Average medium parameters
    u_perp = utilities.dtau_avg(func=lambda x: event.u_perp(point=x, phi=p_phi), point=point, phi=p_phi,
                                dtau=config.jet.DTAU, beta=beta)
    u_tau = utilities.dtau_avg(func=lambda x: event.u_par(point=x, phi=p_phi), point=point, phi=p_phi,
                               dtau=config.jet.DTAU, beta=beta)
    grad_perp_u_perp = utilities.dtau_avg(func=lambda x: event.grad_perp_u_perp(point=x, phi=p_phi),
                                          point=point, phi=p_phi, dtau=config.jet.DTAU, beta=beta)

    return (-1) * (time - event.t0) * (grad_perp_u_perp * (1 / (1-u_tau)))

class num_eloss_interpolator():
    # Instantiation statement. All parameters optional.
    def __init__(self):
        logging.info('Loading numerical energy loss tables...')
        # Find directory of this file
        project_path = os.path.dirname(os.path.realpath(__file__))

        # Load tables of computed fixed length energy loss
        if config.constants.G == 1.6:
            self.g_table = np.load(project_path + '/e_loss_tables/g1.8_deltaE_samples_g_1subdiv.npz')
            self.q_table = np.load(project_path + '/e_loss_tables/g1.8_deltaE_samples_q_1subdiv.npz')
        elif config.constants.G == 1.7:
            self.g_table = np.load(project_path + '/e_loss_tables/g1.8_deltaE_samples_g_1subdiv.npz')
            self.q_table = np.load(project_path + '/e_loss_tables/g1.8_deltaE_samples_q_1subdiv.npz')
        elif config.constants.G == 1.8:
            self.g_table = np.load(project_path + '/e_loss_tables/g1.8_deltaE_samples_g_1subdiv.npz')
            self.q_table = np.load(project_path + '/e_loss_tables/g1.8_deltaE_samples_q_1subdiv.npz')
        elif config.constants.G == 1.9:
            self.g_table = np.load(project_path + '/e_loss_tables/g1.9_deltaE_samples_g_1subdiv.npz')
            self.q_table = np.load(project_path + '/e_loss_tables/g1.9_deltaE_samples_q_1subdiv.npz')
        elif config.constants.G == 2:
            self.g_table = np.load(project_path + '/e_loss_tables/g2.0_deltaE_samples_g_1subdiv.npz')
            self.q_table = np.load(project_path + '/e_loss_tables/g2.0_deltaE_samples_q_1subdiv.npz')
        elif config.constants.G == 2.1:
            self.g_table = np.load(project_path + '/e_loss_tables/g2.1_deltaE_samples_g_1subdiv.npz')
            self.q_table = np.load(project_path + '/e_loss_tables/g2.1_deltaE_samples_q_1subdiv.npz')
        elif config.constants.G == 2.2:
            self.g_table = np.load(project_path + '/e_loss_tables/g2.1_deltaE_samples_g_1subdiv.npz')
            self.q_table = np.load(project_path + '/e_loss_tables/g2.1_deltaE_samples_q_1subdiv.npz')
        elif config.constants.G < 2.2 and config.constants.G > 1.6:
            # Future: Interpolate between
            logging.error('No suitable energy loss table!!!')
        else:
            logging.error('No suitable energy loss table!!!')
            raise Exception

        # Compute pathlength gradient to get energy loss rate tables
        g_delta_E_grad_L = np.gradient(self.g_table['delta_E_vals'], self.g_table['L_points'], axis=2)
        q_delta_E_grad_L = np.gradient(self.q_table['delta_E_vals'], self.q_table['L_points'], axis=2)

        # Interpolate energy loss rate tables
        self.g_dE_dx = RegularGridInterpolator(  # gluons
            (self.g_table['E_points'],
             self.g_table['T_points'],
             self.g_table['L_points']),
             g_delta_E_grad_L,
             bounds_error=False,  # Do not fail if out of data bounds
             fill_value=None)  # Extrapolate energy loss rate, if necessary

        self.q_dE_dx = RegularGridInterpolator(  # light quarks
            (self.q_table['E_points'],
             self.q_table['T_points'],
             self.q_table['L_points']),
            q_delta_E_grad_L,
            bounds_error=False,  # Do not fail if out of data bounds
            fill_value=None)  # Extrapolate energy loss rate, if necessary

    # Method to return the energy loss rate from finite bound first order GLV
    # emitted gluon k on [mu, np.min([2 * E * x, 2 * E * np.sqrt(x * (1 - x))])],
    # medium gluon q on [0, np.sqrt(3 * mu * E)]
    def eloss_rate(self, event, parton, time):
        # Get parton energy, coordinates, etc.
        E = parton.p_T()
        beta = parton.beta()
        point = parton.coords3(time=time)
        p_rho, p_phi = parton.polar_mom_coords()

        # Get medium properties averaged over timestep
        T = utilities.dtau_avg(func=event.temp, point=point, phi=p_phi, dtau=config.jet.DTAU, beta=beta)
        L = (2*(time - event.t0) + config.jet.DTAU)/2

        # Return energy loss rate for appropriate identity
        # Note minus sign - positive values in table correspond to energy loss
        if parton.part == 'g':
            part = 'g'
            return (-1) * float(self.g_dE_dx(np.array([E, T, L])))
        else:
            part = 'q'
            return (-1) * float(self.q_dE_dx(np.array([E, T, L])))

# Integrand for energy loss
# https://journals.aps.org/prd/pdf/10.1103/PhysRevD.44.R2625
def coll_energy_loss_integrand(event, parton, time):
    FmGeV = 1/0.19732687
    nf = 2  # Source?

    # Get parton coordinates
    point = parton.coords3(time=time)
    p_rho, p_phi = parton.polar_mom_coords()
    E = parton.p_T()
    beta = parton.beta()

    # Average medium parameters
    T = utilities.dtau_avg(func=event.temp, point=point, phi=p_phi, dtau=config.jet.DTAU, beta=beta)
    # Set C_R, "quadratic Casimir of the representation R of SU(3) for the parton"
    if parton.part == 'g':
        # For a gluon it's the adjoint representation C_A = N_c = 3
        CR = 3
    else:
        # For a quark it's the fundamental representation C_F = 4/3 in QCD
        CR = 4/3

    # Set alpha_s
    ALPHAS = (config.constants.G**2) / (4*np.pi)

    # Calculate and return energy loss per unit length of this step.
    mg = (config.constants.G * T / np.sqrt(3)) * np.sqrt(1 + (nf / 6))  # Thermal gluon mass, see paper
    return (-1) * FmGeV * CR * (3 / 4) * (8 * np.pi * (ALPHAS ** 2) / 3) * (1 + (nf / 6)) * (T ** 2) * np.log(
        (2 ** (nf / (2 * (6 + nf)))) * 0.920 * (np.sqrt(E * T) / mg))

