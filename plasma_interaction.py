import scipy as sp
from scipy.integrate import quad
from config import TEMP_CUTOFF  # Cutoff temp for plasma integral


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
def temp_cut(event, jet, time):
    if event.temp(jet.coords3(time=time)) < TEMP_CUTOFF:
        return False
    else:
        return True


# Function to calculate moment given initial conditions & interpolating functions
def moment_integral(event, jet, k=0):

    # We scale the integrand by this multiplier, then remove it after the integration to avoid round-off error
    # dealing with small numbers. Currently not in use.
    fudge_scalar = 1

    # Define integrand - ONLY CORRECT FOR K=0 !!!!!!!!!!!
    def integrand(event, jet, k=0):
        return lambda t: fudge_scalar * ((event.i_int_factor(jet=jet, time=t)[0])
                          * (event.u_perp(jet=jet, time=t) / (1 - event.u_par(jet=jet, time=t)))
                          * (event.mu(jet=jet, time=t) ** (k + 2))
                          * event.rho(jet=jet, time=t)
                          * event.sigma(jet=jet, time=t)) if pos_cut(event=event, jet=jet, time=t) \
                                                             and time_cut(event=event, time=t) and \
                                                             temp_cut(event=event, jet=jet, time=t) else 0

    # Calculate moment point
    print('Evaluating moment integral...')
    raw_quad = sp.integrate.quad(integrand(event=event, jet=jet, k=k), event.t0, event.tf, limit=200)

    # Tack constants on
    # The FERMItoGeV factor of ~5 converts unit factor of fm from line integration over fm to GeV
    FERMItoGeV = (1 / 0.19732687)
    moment = FERMItoGeV * (1 / jet.energy) * (1/fudge_scalar) * raw_quad[0]

    # Error on moment
    moment_error = FERMItoGeV * (1 / jet.energy) * (1/fudge_scalar) * raw_quad[1]  # Not including I(k) error

    #print('Quad: ' + str(moment) + ', +/- ' + str(moment_error))

    return moment, moment_error  # Error on the integral I is something to consider.




