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
    We select the appropriate cross-section for a known jet parton and known medium parton specified when called
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
def inv_lambda(event, jet, point):
    """
    We apply a reciprocal summation between the cross-section times density for a medium gluon and for a medium quark
    to get the mean free path as in https://inspirehep.net/literature/1725162
    """

    return (sigma(event, jet, point, med_parton='g') * event.rho(point, med_parton='g')
              + sigma(event, jet, point, med_parton='q') * event.rho(point, med_parton='q'))

# Define integrand for mean q_drift (k=0 moment)
def jet_drift_integrand(event, jet, time):
    jet_point = jet.coords3(time=time)
    jet_p_rho, jet_p_phi = jet.polar_mom_coords()
    FERMItoGeV = (1 / 0.19732687)  # Source link? -- Converts factor of fermi from integral to factor of GeV^{-1}
    return (FERMItoGeV * (1 / jet.p_T()) * config.constants.K_DRIFT
           * ((event.i_int_factor(jet=jet, point=jet_point))
              * (event.u_perp(point=jet_point, phi=jet_p_phi) / (1 - event.u_par(point=jet_point, phi=jet_p_phi)))
              * (event.mu(point=jet_point)**2)
              * inv_lambda(event=event, jet=jet, point=jet_point)))


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


# Integrand for parameterized energy loss over coupling
# Note - includes coupling constant to approximate scale of
# nuclear modification factor (R_AA) from data
def energy_loss_integrand(event, jet, time, tau, model='BBMG', mean_el_rate=0):
    jet_point = jet.coords3(time=time)
    jet_p_phi = jet.polar_mom_coords()[1]
    FERMItoGeV = (1 / 0.19732687)

    # Select energy loss model and return appropriate energy loss
    if model == 'BBMG':
        # Note that we apply FERMItoGeV twice... Once for the t factor, once for the (int dt).
        return (config.constants.K_BBMG * (-1) * FERMItoGeV ** 2 * tau * event.temp(jet_point) ** 3
                * zeta(q=-1) * (1 / np.sqrt(1 - event.vel(point=jet_point)**2))
                * (1))
    elif model == 'Vitev_hack':
        # Note that we do not apply FERMItoGeV, since this rate is in GeV / fm
        return sample_eloss_rate(mean_rate=mean_el_rate, num_samples=None)
    elif model == 'GLV_hack':
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
        alphas = 0.3

        # Calculate and return energy loss of this step.
        return (CR * alphas / 2) * ((-1) * config.constants.K_BBMG
                                    * (FERMItoGeV ** 2)
                                    * (tau - event.t0)
                                    * (event.mu(point=jet_point)**2)
                                    * inv_lambda(event=event, jet=jet, point=jet_point)
                                    * np.log(jet.p_T0/event.mu(point=jet_point)))
    else:
        return 0





