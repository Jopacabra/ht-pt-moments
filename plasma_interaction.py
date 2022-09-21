import numpy as np
import scipy as sp
from scipy.integrate import quad
import utilities
import logging

# Check and interpret desired percent error.
percent_error = 0.01
relative_error = percent_error*0.01


# Functions that set the bounds of interaction with a grid
# Set time at which plasma integral will begin to return 0.
def time_cut(event, time):
    # NOTE: This is really valuable... Gets max time of interpolated function
    if time > event.tf:
        return False
    else:
        return True


# Set position at which plasma integral will begin to return 0.
def pos_cut(event, jet, time):
    if jet.xpos(time=time) > event.xmax or jet.ypos(time=time) > event.ymax:
        return False
    elif jet.xpos(time=time) < event.xmin or jet.ypos(time=time) < event.ymin:
        return False
    else:
        return True


# Set temperature at which plasma integral will begin to return 0.
def temp_cut(event, jet, time, minTemp=0, maxTemp = 1000):
    currentTemp = event.temp(jet.coords3(time=time))
    if currentTemp < minTemp or currentTemp > maxTemp:
        return False
    else:
        return True


# Define integrand - ONLY CORRECT FOR k=0 !!!!!!!!!!!
def jet_drift_integrand(event, jet, k=0, minTemp=0, maxTemp=1000):
    FERMItoGeV = (1 / 0.19732687)
    return lambda t: FERMItoGeV * (1 / jet.energy) *  ((event.i_int_factor(jet=jet, time=t)[0])
                      * (event.u_perp(jet=jet, time=t) / (1 - event.u_par(jet=jet, time=t)))
                      * (event.mu(jet=jet, time=t) ** (k + 2))
                      * event.rho(jet=jet, time=t)
                      * event.sigma(jet=jet, time=t)) if pos_cut(event=event, jet=jet, time=t) \
                                                         and time_cut(event=event, time=t) and \
                                                         temp_cut(event=event, jet=jet, time=t, minTemp=minTemp,
                                                                  maxTemp=maxTemp) else 0


# Function to calculate moment given initial conditions & interpolating functions
def jet_drift_moment(event, jet, k=0, minTemp=0, maxTemp=1000, quiet=False):

    # Calculate moment point
    if not quiet:
        logging.info('Evaluating jet drift integral...')
    raw_quad = sp.integrate.quad(jet_drift_integrand(event=event, jet=jet, k=k, minTemp=minTemp, maxTemp=maxTemp),
                                 event.t0, event.tf, limit=200, epsrel=relative_error)

    # Tack constants on
    # The FERMItoGeV factor of ~5 converts unit factor of fm from line integration over fm to GeV

    moment = raw_quad[0]

    # Error on moment
    moment_error = raw_quad[1]  # Not including I(k) error

    #print('Quad: ' + str(moment) + ', +/- ' + str(moment_error))

    return moment, moment_error  # Error on the integral I is something to consider.


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
def energy_loss_integrand(event, jet, minTemp=0, maxTemp=1000):
    FERMItoGeV = (1 / 0.19732687)  # Note that we apply this twice... Once for the t factor, once for the (int dt).
    return lambda t: ( (-1) * FERMItoGeV**2 * t * event.temp_jet(jet=jet, time=t)**3
                      * zeta(q=-1) * (1 / np.sqrt(1 - event.vel(jet=jet, time=t)**2))
                      * (1 - event.vel(jet=jet, time=t) * np.cos(jet.theta0 - event.vel_angle(jet=jet, time=t)))
                      ) if pos_cut(event=event, jet=jet, time=t) \
                                                and time_cut(event=event, time=t) and \
                                                temp_cut(event=event, jet=jet, time=t, minTemp=minTemp,
                                                         maxTemp=maxTemp) else 0


# Function to calculate moment given initial conditions & interpolating functions
# Note that this returns the energy loss over the jet-medium coupling. Essentially this means we need to
# fix an overall normalization to fit a data point somewhere in the most precise regime of BBMG model accuracy.
def energy_loss_moment(event, jet, minTemp=0, maxTemp=1000, quiet=False):

    # Calculate moment point
    if not quiet:
        logging.info('Evaluating jet energy loss integral...')
    raw_quad = sp.integrate.quad(energy_loss_integrand(event=event, jet=jet, minTemp=minTemp, maxTemp=maxTemp),
                                 event.t0, event.tf, limit=200, epsrel=relative_error)

    # Tack constants on
    # The FERMItoGeV factor of ~5 converts unit factor of fm from line integration over fm to GeV

    e_loss = raw_quad[0]

    # Error on moment
    e_loss_error = raw_quad[1]

    return e_loss, e_loss_error




