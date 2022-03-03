import numpy as np
import scipy as sp
from scipy.integrate import quad


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


# Function to find minimum x bound of grid
def grid_y_min(temp_func):
    return np.amin(temp_func.grid[2])


# Parameterization of path in terms of time t from initial conditions
# Returns x-coordinate of a jet for given parameters
def x_pos(t, X0, THETA0, V0=1, t_naut=0.5):
    return X0 + (V0 * np.cos(THETA0) * (t - t_naut))


# Parameterization of path in terms of time t from initial conditions
# Returns y-coordinate of a jet for given parameters
def y_pos(t, Y0, THETA0, V0=1, t_naut=0.5):
    return Y0 + (V0 * np.cos(THETA0) * (t - t_naut))


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


# Function to calculate moment given initial conditions & interpolating functions
def moment_integral(temp_func, x_vel, y_vel, X0, Y0, THETA0, K=0, G=2, JET_E=10 ** 9, V0=1, tempCutoff=0):
    # Determine final time for integral

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
    def temp_cut(temp_func, t, X0, Y0, THETA0, V0=1):
        if temp(temp_func, t, X0, Y0, THETA0, V0=V0) < tempCutoff:
            return False
        else:
            return True

    # Define Debye mass and density
    def rho(temp_func, t, X0, Y0, THETA0, V0=1):
        return 1.202056903159594 * 16 * (1 / (np.pi ** 2)) \
               * temp(temp_func, t, X0, Y0, THETA0,
                      V0=V0) ** 3  # Chosen to be ideal gluon gas dens. as per Sievert, Yoon, et. al.

    def mu(temp_func, t, X0, Y0, THETA0, V0=1):
        return G * temp(temp_func, t, X0, Y0, THETA0, V0=V0)

    def sigma(temp_func, t, X0, Y0, THETA0, V0=1):
        return np.pi * G ** 4 / (mu(temp_func, t, X0, Y0, THETA0, V0=V0) ** 2)

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
            temp_func, t, X0, Y0, THETA0, V0=V0) else 0

    # Calculate moment point
    print('Evaluating moment integral...')
    raw_quad = sp.integrate.quad(integrand(temp_func, X0, Y0, THETA0, V0=V0, K=K), t_naut(temp_func), t_final(temp_func))
    # Tack constants on
    moment = (1 / JET_E) * raw_quad[0]
    # Error on moment
    moment_error = (1 / JET_E) * raw_quad[1]  # Not including I(k) error

    return moment, moment_error


# Function to calculate moment given initial conditions & interpolating functions
def moment_integral_legacy(temp_raw, x_vel, y_vel, X0, Y0, THETA0, K):
    # Set integration constants and such
    G = 2
    JET_E = 10 ** 9  # Jet energy in GeV
    V0 = 1

    # Determine final time for integral
    t_naut = np.amin(temp_raw.grid[0])
    t_final = np.amax(temp_raw.grid[0])

    # Parameterize path in terms of t from initial conditions
    def x_pos(t):
        return X0 + (V0 * np.cos(THETA0) * (t - t_naut))

    def y_pos(t):
        return Y0 + (V0 * np.cos(THETA0) * (t - t_naut))

    # Set up perp and parallel velocity components
    def u_perp(t):
        return -x_vel(np.array([t, x_pos(t), y_pos(t)])) * np.sin(THETA0) \
               + y_vel(np.array([t, x_pos(t), y_pos(t)])) * np.cos(THETA0)

    def u_par(t):
        return x_vel(np.array([t, x_pos(t), y_pos(t)])) * np.cos(THETA0) \
               + y_vel(np.array([t, x_pos(t), y_pos(t)])) * np.sin(THETA0)

    def temp(t):
        return temp_raw(np.array([t, x_pos(t), y_pos(t)]))

    # Set time, position, and temperature cuts
    def time_cut(t):
        # NOTE: This is really valuable... Gets max time of interpolated function
        if t > np.amax(temp_raw.grid[0]):
            return False
        else:
            return True

    def pos_cut(t):
        if x_pos(t) > np.amax(temp_raw.grid[1]) or y_pos(t) > np.amax(temp_raw.grid[2]):
            return False
        elif x_pos(t) < np.amin(temp_raw.grid[1]) or y_pos(t) < np.amin(temp_raw.grid[2]):
            return False
        else:
            return True

    def temp_cut(t):
        if temp(t) < 0:
            return False
        else:
            return True

    # Define Debye mass and density
    def rho(t):
        return 1.202056903159594 * 16 * (1 / (np.pi ** 2)) \
               * temp(t) ** 3  # Chosen to be ideal gluon gas dens. as per Sievert, Yoon, et. al.

    def mu(t):
        return G * temp(t)

    def sigma(t):
        return np.pi * G ** 4 / (mu(t) ** 2)

    # Set up I(k) business
    def i_integral(t):
        if K == 0:
            return [3 * np.log(JET_E / mu(t)), 0]  # No idea what the error should be here
        elif K == -4:
            i_integrand = lambda x: (x ** (1 + (K / 2))) * ((3 * x + 2) / ((x + 1) ** 3))
            return quad(i_integrand, 0, np.inf)[0]
        else:
            i_integrand = lambda x: (x ** (1 + (K / 2))) * ((3 * x + 2) / ((x + 1) ** 3))
            return quad(i_integrand, 0, np.inf)[0]

    # Define integrand - ONLY CORRECT FOR K=0 !!!!!!!!!!!
    def integrand(t):
        if pos_cut(t) and time_cut(t) and temp_cut(t):
            return (i_integral(t)[0]) * (u_perp(t) / (1 - u_par(t))) * (mu(t) ** (K + 2)) * rho(t) * sigma(t)
        else:
            return 0

    # Calculate moment point
    print('Evaluating moment integral...')
    raw_quad = sp.integrate.quad(integrand, t_naut, t_final)
    # Tack constants on
    moment = (1 / JET_E) * raw_quad[0]
    # Error on moment
    moment_error = raw_quad[1]  # Not including I(k) error

    return moment, moment_error


# Error on the integral I is something to consider. Maxes out around 5 x 10^-8.

def err_demo():
    for i in np.arange(-4, 0, 0.1):
        k = i
        i_integrand = lambda x: (x ** (1 + (k / 2))) * ((3 * x + 2) / ((x + 1) ** 3))
        i_int = quad(i_integrand, 0, np.inf)
        print('k = ' + str(k) + ': ' + str(i_int))
