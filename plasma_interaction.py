import scipy as sp
from scipy.integrate import quad

# Check and interpret desired percent error.
percent_error = 0.01
relative_error = percent_error*0.01

# We scale the integrand by this multiplier, then remove it after the integration to avoid round-off error
# dealing with small numbers.
fudge_scalar = 1


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
def integrand(event, jet, k=0, minTemp=0, maxTemp=1000):
    FERMItoGeV = (1 / 0.19732687)
    return lambda t: fudge_scalar * FERMItoGeV * (1 / jet.energy) *  ((event.i_int_factor(jet=jet, time=t)[0])
                      * (event.u_perp(jet=jet, time=t) / (1 - event.u_par(jet=jet, time=t)))
                      * (event.mu(jet=jet, time=t) ** (k + 2))
                      * event.rho(jet=jet, time=t)
                      * event.sigma(jet=jet, time=t)) if pos_cut(event=event, jet=jet, time=t) \
                                                         and time_cut(event=event, time=t) and \
                                                         temp_cut(event=event, jet=jet, time=t, minTemp=minTemp,
                                                                  maxTemp=maxTemp) else 0


# Function to calculate moment given initial conditions & interpolating functions
def moment_integral(event, jet, k=0, minTemp=0, maxTemp=1000, quiet=False):

    # Calculate moment point
    if not quiet:
        print('Evaluating moment integral...')
    raw_quad = sp.integrate.quad(integrand(event=event, jet=jet, k=k, minTemp=minTemp, maxTemp=maxTemp), event.t0, event.tf, limit=200
                                 , epsrel=relative_error)

    # Tack constants on
    # The FERMItoGeV factor of ~5 converts unit factor of fm from line integration over fm to GeV

    moment = (1/fudge_scalar) * raw_quad[0]

    # Error on moment
    moment_error = (1/fudge_scalar) * raw_quad[1]  # Not including I(k) error

    #print('Quad: ' + str(moment) + ', +/- ' + str(moment_error))

    return moment, moment_error  # Error on the integral I is something to consider.




