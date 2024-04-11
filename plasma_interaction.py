import numpy as np
import config
import utilities
import logging

# Check and interpret desired percent error.
percent_error = 0.01
relative_error = percent_error*0.01

# Hacky sampling of a new energy loss rate.
def sample_eloss_rate(mean_rate, num_samples=None):
    # Create random number generator instance
    rng = np.random.default_rng()

    # Sample an energy loss rate from "continuous Poisson" we approximate as a gamma dist with this mean and scale of 1.
    eloss_rate = rng.gamma(shape=mean_rate, scale=1.0, size=num_samples)

    if eloss_rate < 0:
        eloss_rate = 0

    return eloss_rate

# Function to return total cross section at a particular point for jet parton and *gluon* in medium
# Total GW cross section, as per Sievert, Yoon, et. al.
# Specify med_parton either 'g' for medium gluon or 'q' for generic light (?) quark in medium
# https://inspirehep.net/literature/1725162
def sigma(event, jet, point, med_parton='g'):
    """
    We select the appropriate cross-section for a known jet parton and
    known medium parton specified when called
    """
    current_point = point
    coupling = config.constants.G

    if (jet.part == 'u' or jet.part == 'ubar' or jet.part == 'd' or jet.part == 'dbar' or jet.part == 's'
            or jet.part == 'sbar'):
        jet_parton = 'q'
    elif jet.part == 'g':
        jet_parton = 'g'
    else:
        jet_parton = None

    sigma_gg_gg = (9/(32 * np.pi)) * coupling ** 4 / (event.mu(point=current_point) ** 2)
    sigma_qg_qg = (1/(8 * np.pi)) * coupling ** 4 / (event.mu(point=current_point) ** 2)
    sigma_qq_qq = (1/(18 * np.pi)) * coupling ** 4 / (event.mu(point=current_point) ** 2)

    if jet_parton == 'g' and med_parton == 'g':
        # gg -> gg cross-section
        cross_section = sigma_gg_gg
    elif jet_parton == 'q' and med_parton == 'g':
        # qg -> qg cross-section
        cross_section = sigma_qg_qg
    elif jet_parton == 'g' and med_parton == 'q':
        # qg -> qg cross-section
        cross_section = sigma_qg_qg
    elif jet_parton == 'q' and med_parton == 'q':
        # qq -> qq cross-section
        cross_section = sigma_qq_qq
    else:
        logging.debug('Unknown parton scattering cs... Using gg->gg scattering cross section')
        cross_section = sigma_gg_gg

    return cross_section

# Function to return inverse QGP drift mean free path in units of GeV^{-1}
# Total GW cross section, as per Sievert, Yoon, et. al.
def inv_lambda(event, jet, point, med_parton='all'):
    """
    We apply a reciprocal summation between the cross-section times density for a medium gluon and for a medium quark
    to get the mean free path as in https://inspirehep.net/literature/1725162
    """

    if med_parton == 'all':
        return (sigma(event, jet, point, med_parton='g') * event.rho(point, med_parton='g')
              + sigma(event, jet, point, med_parton='q') * event.rho(point, med_parton='q'))
    else:
        return sigma(event, jet, point, med_parton=med_parton) * event.rho(point, med_parton=med_parton)

# Define integrand for mean q_drift (k=0 moment)
def jet_drift_integrand(event, jet, time):
    jet_point = jet.coords3(time=time)
    jet_p_rho, jet_p_phi = jet.polar_mom_coords()
    FmGeV = 0.19732687
    # Source link? -- Converts factor of fermi from integral to factor of GeV^{-1}
    return ((1 / FmGeV) * (1 / jet.p_T()) * config.jet.K_F_DRIFT
           * ((event.i_int_factor(jet=jet, point=jet_point))
              * (event.u_perp(point=jet_point, phi=jet_p_phi) / (1 - event.u_par(point=jet_point, phi=jet_p_phi)))
              * (event.mu(point=jet_point)**2)
              * inv_lambda(event=event, jet=jet, point=jet_point)))

# Define integrand for mean flow-grad drift
def flowgrad_drift_integrand(event, jet, time):
    jet_point = jet.coords3(time=time)
    jet_p_rho, jet_p_phi = jet.polar_mom_coords()
    FmGeV = 0.19732687
    T = event.temp(jet_point)
    uperp = event.u_perp(point=jet_point, phi=jet_p_phi)
    upar = event.u_par(point=jet_point, phi=jet_p_phi)
    grad_perp_temp = event.grad_perp_T(jet_point, jet_p_phi)
    grad_perp_u_perp = event.grad_perp_u_perp(jet_point, jet_p_phi)
    grad_perp_u_tau = event.grad_perp_u_par(jet_point, jet_p_phi)
    g = config.constants.G
    pt = jet.p_T()
    # Source link? -- Converts factor of fermi from integral to factor of GeV^{-1}
    return - ((1 / FmGeV) * (g**2 / pt) * config.jet.K_FG_DRIFT * (3 / 2)
        * (time - event.t0) * inv_lambda(event=event, jet=jet, point=jet_point)
        * ((grad_perp_temp) * (uperp/((1 - upar)**2)) * (3 * T * np.log(pt / (g * T)) - T )
        + grad_perp_u_tau * (2/((1 - upar)**3))  * uperp * (T**2) * np.log(pt / (g * T))
        + grad_perp_u_perp * (2 * uperp/((1 - upar)**2)) * (T**2) * np.log(pt / (g * T))))

# Define integrand for mean flow-grad_uT drift
def flowgrad_T_integrand(event, jet, time):
    jet_point = jet.coords3(time=time)
    jet_p_rho, jet_p_phi = jet.polar_mom_coords()
    FmGeV = 0.19732687
    T = event.temp(jet_point)
    uperp = event.u_perp(point=jet_point, phi=jet_p_phi)
    utau = event.u_par(point=jet_point, phi=jet_p_phi)
    grad_perp_temp = event.grad_perp_T(jet_point, jet_p_phi)
    mu = event.mu(point=jet_point)
    E = jet.p_T()

    return - ((1 / FmGeV) * (3 / E) * config.jet.K_FG_DRIFT * (time - event.t0)
              * 3 * grad_perp_temp * ((uperp**2)/((1 - utau)**2)) * (1/T)
              * (mu**2) * inv_lambda(event=event, jet=jet, point=jet_point)
              * np.log(E / mu))

# Define integrand for mean flow-grad_utau drift
def flowgrad_utau_integrand(event, jet, time):
    jet_point = jet.coords3(time=time)
    jet_p_rho, jet_p_phi = jet.polar_mom_coords()
    FmGeV = 0.19732687
    T = event.temp(jet_point)
    uperp = event.u_perp(point=jet_point, phi=jet_p_phi)
    utau = event.u_par(point=jet_point, phi=jet_p_phi)
    grad_perp_u_tau = event.grad_perp_u_par(jet_point, jet_p_phi)
    mu = event.mu(point=jet_point)
    E = jet.p_T()
    # Source link? -- Converts factor of fermi from integral to factor of GeV^{-1}
    return - ((1 / FmGeV) * (3 / E) * config.jet.K_FG_DRIFT * (time - event.t0)
              * 2 * grad_perp_u_tau * ((uperp**2)/((1 - utau)**3))
              * (mu**2) * inv_lambda(event=event, jet=jet, point=jet_point)
              * np.log(E / mu))

# Define integrand for mean flow-grad_uperp drift
def flowgrad_uperp_integrand(event, jet, time):
    jet_point = jet.coords3(time=time)
    jet_p_rho, jet_p_phi = jet.polar_mom_coords()
    FmGeV = 0.19732687
    T = event.temp(jet_point)
    uperp = event.u_perp(point=jet_point, phi=jet_p_phi)
    utau = event.u_par(point=jet_point, phi=jet_p_phi)
    grad_perp_u_perp = event.grad_perp_u_perp(jet_point, jet_p_phi)
    mu = event.mu(point=jet_point)
    E = jet.p_T()
    # Source link? -- Converts factor of fermi from integral to factor of GeV^{-1}
    return - ((1 / FmGeV) * (3 / E) * config.jet.K_FG_DRIFT * (time - event.t0)
              * 2 * grad_perp_u_perp * (uperp/((1 - utau)**2))
              * (mu**2) * inv_lambda(event=event, jet=jet, point=jet_point)
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
def energy_loss_integrand(event, jet, time, tau, model='BBMG', fgqhat=False, mean_el_rate=0):
    jet_point = jet.coords3(time=time)
    jet_p_phi = jet.polar_mom_coords()[1]
    FmGeV = 0.19732687

    # Select energy loss model and return appropriate energy loss
    if model == 'BBMG':
        # Note that we apply FERMItoGeV twice... Once for the t factor, once for the (int dt).
        return (config.jet.K_BBMG * (-1) * ((1 / FmGeV) ** 2) * time * event.temp(jet_point) ** 3
                * zeta(q=-1) * (1 / np.sqrt(1 - event.vel(point=jet_point)**2))
                * (1))
    elif model == 'Vitev_hack':
        # Note that we do not apply FERMItoGeV, since this rate is in GeV / fm
        return (-1) * sample_eloss_rate(mean_rate=mean_el_rate, num_samples=None)
    elif model == 'GLV':
        # https://inspirehep.net/literature/539404
        # Note that we apply FERMItoGeV twice... Once for the t factor, once for the (int dt).
        # Set C_R, "quadratic Casimir of the representation R of SU(3) for the jet"
        if jet.part == 'g':
            # For a gluon it's the adjoint representation C_A = N_c = 3
            CR = 3
        else:
            # For a quark it's the fundamental representation C_F = 4/3 in QCD
            CR = 4/3

        # Set alpha_s
        # "For alpha_s the scale runs, but ballpark you can guess 0.3" - Dr. Sievert
        alphas = (config.constants.G**2) / (4*np.pi)

        # Calculate and return energy loss per unit length of this step.
        return (-1)*(CR * alphas / 2) * (((1 / FmGeV) ** 2)
                                    * (time - event.t0)
                                    * (event.mu(point=jet_point)**2)
                                    * inv_lambda(event=event, jet=jet, point=jet_point)
                                    * np.log(jet.p_T()/event.mu(point=jet_point)))
    else:
        return 0

# Integrand for gradient deflection to 2nd order in opacity
# Note - first moment is zero. Essentially computing cuberoot(q_{grad}^3) as scale approx.
def grad_integrand(event, jet, time, tau):
    jet_point = jet.coords3(time=time)
    jet_p_rho, jet_p_phi = jet.polar_mom_coords()
    FmGeV = 0.19732687

    '''
    Omega here is the characteristic width of the gaussian approximating the jet width spectrum.
    For a simple first investigation of the order of magnitude of the gradient effects, we 
    assume that this is equivalent to the gluon saturation scale in a p-X collision system.
    
    We take the gluon saturation scale from the pocket equation in Eq. 3.6 here:
    https://inspirehep.net/literature/1206324
    fit by eye to the results in Fig. 3.9 left for Au at x = 0.0001, which roughly equates to 
    the region of 1 GeV jets in Au Au collisions at sqrt(s) == 5.02 TeV
    '''

    # Select proper saturation scale from scaled pocket equation
    if config.transport.trento.PROJ1 == 'Pb' and config.transport.trento.PROJ2 == 'Pb':
        A = 208
    else:
        A = 197
    x = jet.p_T() / config.constants.ROOT_S
    omega = 0.01675 * ((A / x) ** (1 / 3))

    first_order_q = FmGeV*(((2 * (omega**2) * tau * (event.mu(point=jet_point)**2)
                       * event.grad_perp_rho(point=jet_point, phi=jet_p_phi, med_parton='q')
                    * inv_lambda(event=event, jet=jet, point=jet_point, med_parton='q'))
                   / (jet.p_T() * event.rho(jet_point, med_parton='q')))
                   * np.log(jet.p_T()/event.mu(point=jet_point)))

    first_order_g = FmGeV*(((2 * (omega**2) * tau * (event.mu(point=jet_point)**2)
                       * event.grad_perp_rho(point=jet_point, phi=jet_p_phi, med_parton='g')
                    * inv_lambda(event=event, jet=jet, point=jet_point, med_parton='g'))
                   / (jet.p_T() * event.rho(jet_point, med_parton='g')))
                   * np.log(jet.p_T()/event.mu(point=jet_point)))

    second_order_q = (FmGeV**2) * ((tau**2) * (event.mu(point=jet_point)**4) * event.grad_perp_rho(point=jet_point, phi=jet_p_phi, med_parton='q')
                      * (inv_lambda(event=event, jet=jet, point=jet_point, med_parton='q')**2)
                      * (np.log(jet.p_T()/event.mu(point=jet_point))**2)
                    / (2 * jet.p_T() * (event.rho(jet_point, med_parton='q'))))

    second_order_g = (FmGeV**2) * ((tau ** 2) * (event.mu(point=jet_point) ** 4) * event.grad_perp_rho(point=jet_point, phi=jet_p_phi, med_parton='g')
                * (inv_lambda(event=event, jet=jet, point=jet_point, med_parton='g') ** 2)
                * (np.log(jet.p_T() / event.mu(point=jet_point)) ** 2)
                / (2 * jet.p_T() * (event.rho(jet_point, med_parton='g'))))

    return np.cbrt(first_order_q + first_order_g + second_order_q + second_order_g)

# Modification factor for energy loss due to gradients of temperature
def fg_T_qhat_mod_factor(event, jet, time):
    jet_point = jet.coords3(time=time)
    jet_p_rho, jet_p_phi = jet.polar_mom_coords()
    # FmGeV = 0.19732687
    T = event.temp(jet_point)
    uperp = event.u_perp(point=jet_point, phi=jet_p_phi)
    upar = event.u_par(point=jet_point, phi=jet_p_phi)
    grad_perp_temp = event.grad_perp_T(jet_point, jet_p_phi)
    grad_perp_u_perp = event.grad_perp_u_perp(jet_point, jet_p_phi)
    grad_perp_u_tau = event.grad_perp_u_par(jet_point, jet_p_phi)
    # g = config.constants.G
    pt = jet.p_T()
    mu = event.mu(point=jet_point)
    return (-1) * (time - event.t0) * (3 * grad_perp_temp * (uperp / (1-upar)) * (1/T))


# Modification factor for energy loss due to gradients of utau
def fg_utau_qhat_mod_factor(event, jet, time):
    jet_point = jet.coords3(time=time)
    jet_p_rho, jet_p_phi = jet.polar_mom_coords()
    # FmGeV = 0.19732687
    T = event.temp(jet_point)
    uperp = event.u_perp(point=jet_point, phi=jet_p_phi)
    upar = event.u_par(point=jet_point, phi=jet_p_phi)
    grad_perp_temp = event.grad_perp_T(jet_point, jet_p_phi)
    grad_perp_u_perp = event.grad_perp_u_perp(jet_point, jet_p_phi)
    grad_perp_u_tau = event.grad_perp_u_par(jet_point, jet_p_phi)
    # g = config.constants.G
    pt = jet.p_T()
    mu = event.mu(point=jet_point)
    return (-1) * (time - event.t0) * (grad_perp_u_tau * (uperp / ((1-upar)**2)))

# Modification factor for energy loss due to gradients of uperp
def fg_uperp_qhat_mod_factor(event, jet, time):
    jet_point = jet.coords3(time=time)
    jet_p_rho, jet_p_phi = jet.polar_mom_coords()
    # FmGeV = 0.19732687
    T = event.temp(jet_point)
    uperp = event.u_perp(point=jet_point, phi=jet_p_phi)
    upar = event.u_par(point=jet_point, phi=jet_p_phi)
    grad_perp_temp = event.grad_perp_T(jet_point, jet_p_phi)
    grad_perp_u_perp = event.grad_perp_u_perp(jet_point, jet_p_phi)
    grad_perp_u_tau = event.grad_perp_u_par(jet_point, jet_p_phi)
    # g = config.constants.G
    pt = jet.p_T()
    mu = event.mu(point=jet_point)
    return (-1) * (time - event.t0) * (grad_perp_u_perp * (1 / (1-upar)))