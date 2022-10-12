import numpy as np
import scipy as sp
from scipy.integrate import quad
import utilities
import logging

# Check and interpret desired percent error.
percent_error = 0.01
relative_error = percent_error*0.01

# Define integrand - ONLY CORRECT FOR k=0 !!!!!!!!!!!
def jet_drift_integrand(event, jet, time):
    jet_point = jet.coords3(time=time)
    jet_p_rho, jet_p_phi = jet.polar_mom_coords()
    FERMItoGeV = (1 / 0.19732687)
    return FERMItoGeV * (1 / jet.p_T()) * ((event.i_int_factor(point=jet_point, jet_pT=jet.p_T()))
                                                   * (event.u_perp(point=jet_point, phi=jet_p_phi) /
                                                      (1 - event.u_par(point=jet_point, phi=jet_p_phi)))
                                                   * (event.mu(point=jet_point)**2)
                                                   * event.rho(point=jet_point)
                                                   * event.sigma(point=jet_point))


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
def energy_loss_integrand(event, jet, time, tau):
    jet_point = jet.coords3(time=time)
    jet_p_phi = jet.polar_mom_coords()[1]
    FERMItoGeV = (1 / 0.19732687)  # Note that we apply this twice... Once for the t factor, once for the (int dt).
    return ( (-1) * FERMItoGeV**2 * tau * event.temp(jet_point)**3
                      * zeta(q=-1) * (1 / np.sqrt(1 - event.vel(point=jet_point)**2))
                      * (1 - event.vel(point=jet_point) * np.cos(jet_p_phi
                                                                 - event.vel_angle(point=jet_point))))




