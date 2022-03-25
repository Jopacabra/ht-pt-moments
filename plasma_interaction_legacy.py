import numpy as np
import scipy as sp
from scipy.integrate import quad

# FUTURE THREADING!!!!!!!
G = 2

# Function to find initial time of a given interpolated grid
def t_naut(temp_func):
    return np.amin(temp_func.grid[0])


# Function to find initial time of a given interpolated grid
def t_final(temp_func):
    return np.amax(temp_func.grid[0])


# Function to find maximum x bound of grid
def grid_x_max(temp_func):
    return np.amax(temp_func.grid[1])


# Function to find minimum x bound of grid
def grid_x_min(temp_func):
    return np.amin(temp_func.grid[1])


# Function to find maximum y bound of grid
def grid_y_max(temp_func):
    return np.amax(temp_func.grid[2])


# Function to find minimum y bound of grid
def grid_y_min(temp_func):
    return np.amin(temp_func.grid[2])


# Parameterization of path in terms of time t from initial conditions
# Returns x-coordinate of a jet for given parameters
def x_pos(t, X0, THETA0, V0=1, t_naut=0.5):
    return X0 + (V0 * np.cos(THETA0) * (t - t_naut))


# Parameterization of path in terms of time t from initial conditions
# Returns y-coordinate of a jet for given parameters
# Recall fix - this is sine function
def y_pos(t, Y0, THETA0, V0=1, t_naut=0.5):
    return Y0 + (V0 * np.sin(THETA0) * (t - t_naut))


# Returns component of medium velocity perpendicular to the jet axis
def u_perp(temp_func, x_vel, y_vel, t, X0, Y0, THETA0, V0=1):
    return -x_vel(np.array([t, x_pos(t, X0, THETA0, V0=V0, t_naut=t_naut(temp_func)), y_pos(t, Y0, THETA0, V0=V0, t_naut=t_naut(temp_func))])) * np.sin(THETA0) \
           + y_vel(np.array([t, x_pos(t, X0, THETA0, V0=V0, t_naut=t_naut(temp_func)), y_pos(t, Y0, THETA0, V0=V0, t_naut=t_naut(temp_func))])) * np.cos(THETA0)


# Returns component of medium velocity parallel to the jet axis
def u_par(temp_func, x_vel, y_vel, t, X0, Y0, THETA0, V0=1):
    return x_vel(np.array([t, x_pos(t, X0, THETA0, V0=V0, t_naut=t_naut(temp_func)), y_pos(t, Y0, THETA0, V0=V0, t_naut=t_naut(temp_func))])) * np.cos(THETA0) \
           + y_vel(np.array([t, x_pos(t, X0, THETA0, V0=V0, t_naut=t_naut(temp_func)), y_pos(t, Y0, THETA0, V0=V0, t_naut=t_naut(temp_func))])) * np.sin(THETA0)

# Returns magnitude of medium velocity
def vel_mag(temp_func, x_vel, y_vel, t, X0, Y0, THETA0, V0=1):
    return np.sqrt(x_vel(np.array([t, x_pos(t, X0, THETA0, V0=V0, t_naut=t_naut(temp_func)), y_pos(t, Y0, THETA0, V0=V0, t_naut=t_naut(temp_func))]))**2
                   + y_vel(np.array([t, x_pos(t, X0, THETA0, V0=V0, t_naut=t_naut(temp_func)), y_pos(t, Y0, THETA0, V0=V0, t_naut=t_naut(temp_func))])) **2)


def temp(temp_func, t, X0, Y0, THETA0, V0=1):
    return temp_func(np.array([t, x_pos(t, X0, THETA0, V0=V0, t_naut=t_naut(temp_func)), y_pos(t, Y0, THETA0, V0=V0, t_naut=t_naut(temp_func))]))


# Define Debye mass and density
def rho(temp_func, t, X0, Y0, THETA0, V0=1):
    return 1.202056903159594 * 16 * (1 / (np.pi ** 2)) \
           * temp(temp_func, t, X0, Y0, THETA0,
                  V0=V0) ** 3  # Chosen to be ideal gluon gas dens. as per Sievert, Yoon, et. al.

def mu(temp_func, t, X0, Y0, THETA0, V0=1):
    return G * temp(temp_func, t, X0, Y0, THETA0, V0=V0)

def sigma(temp_func, t, X0, Y0, THETA0, V0=1):
    return np.pi * G ** 4 / (mu(temp_func, t, X0, Y0, THETA0, V0=V0) ** 2) # Total GW cross section, as per Sievert, Yoon, et. al.

def i_int_factor(temp_func, t, X0, Y0, THETA0, V0=1, JET_E=10):
    return 3 * np.log(JET_E / mu(temp_func, t, X0, Y0, THETA0, V0=V0))

# Functions that set the bounds of interaction with a grid
# Set time at which plasma integral will begin to return 0.
def time_cut(temp_func, t):
    # NOTE: This is really valuable... Gets max time of interpolated function
    if t > t_final(temp_func):
        return False
    else:
        return True

# Set position at which plasma integral will begin to return 0.
def pos_cut(temp_func, t, X0, Y0, THETA0, V0=1):
    if x_pos(t, X0, THETA0, V0=V0, t_naut=t_naut(temp_func)) > grid_x_max(temp_func) or y_pos(t, Y0, THETA0, V0=V0, t_naut=t_naut(temp_func)) > grid_y_max(temp_func):
        return False
    elif x_pos(t, X0, THETA0, V0=V0, t_naut=t_naut(temp_func)) < grid_y_min(temp_func) or y_pos(t, Y0, THETA0, V0=V0, t_naut=t_naut(temp_func)) < grid_y_min(temp_func):
        return False
    else:
        return True

# Set temperature at which plasma integral will begin to return 0.
def temp_cut(temp_func, t, X0, Y0, THETA0, V0=1, tempCutoff=0):
    if temp(temp_func, t, X0, Y0, THETA0, V0=V0) < tempCutoff:
        return False
    else:
        return True


# Function to calculate moment given initial conditions & interpolating functions
def moment_integral(temp_func, x_vel, y_vel, X0, Y0, THETA0, K=0, G=2, JET_E=10, V0=1, tempCutoff=0):

    # Set up I(k) business
    def i_integral(temp_func, t, X0, Y0, THETA0, V0=1):
        if K == 0:
            return [3 * np.log(JET_E / mu(temp_func, t, X0, Y0, THETA0, V0=V0)),
                    0]  # No idea what the error should be here
        elif K == -4:
            i_integrand = lambda x: (x ** (1 + (K / 2))) * ((3 * x + 2) / ((x + 1) ** 3))
            return quad(i_integrand, 0, np.inf)[0]
        else:
            i_integrand = lambda x: (x ** (1 + (K / 2))) * ((3 * x + 2) / ((x + 1) ** 3))
            return quad(i_integrand, 0, np.inf)[0]

    # Define integrand - ONLY CORRECT FOR K=0 !!!!!!!!!!!
    def integrand(temp_func, X0, Y0, THETA0, V0=1, K=0):
        return lambda t: ((i_integral(temp_func, t, X0, Y0, THETA0, V0=V0)[0])
                          * (u_perp(temp_func, x_vel, y_vel, t, X0, Y0, THETA0, V0=V0)
                             / (1 - u_par(temp_func, x_vel, y_vel, t, X0, Y0, THETA0, V0=V0)))
                          * (mu(temp_func, t, X0, Y0, THETA0, V0=V0) ** (K + 2))
                          * rho(temp_func, t, X0, Y0, THETA0, V0=V0)
                          * sigma(temp_func, t, X0, Y0, THETA0, V0=V0)) if pos_cut(temp_func, t, X0, Y0, THETA0,
                                                                                   V0=V0) and time_cut(temp_func,
                                                                                                       t) and temp_cut(
            temp_func, t, X0, Y0, THETA0, V0=V0, tempCutoff=tempCutoff) else 0

    # Calculate moment point
    print('Evaluating moment integral...')
    raw_quad = sp.integrate.quad(integrand(temp_func, X0, Y0, THETA0, V0=V0, K=K), t_naut(temp_func), t_final(temp_func))
    # Tack constants on
    FERMItoGeV = (1 / 0.19732687)
    moment = FERMItoGeV * (1 / JET_E) * raw_quad[0] # The FERMItoGeV factor of ~5 converts unit factor of fm from line integration over fm to GeV
    # Error on moment
    moment_error = FERMItoGeV * (1 / JET_E) * raw_quad[1]  # Not including I(k) error

    return moment, moment_error


# Error on the integral I is something to consider.
def err_demo():
    for i in np.arange(-4, 0, 0.1):
        k = i
        i_integrand = lambda x: (x ** (1 + (k / 2))) * ((3 * x + 2) / ((x + 1) ** 3))
        i_int = quad(i_integrand, 0, np.inf)
        print('k = ' + str(k) + ': ' + str(i_int))
