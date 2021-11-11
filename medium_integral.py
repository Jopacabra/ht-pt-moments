import numpy as np
import scipy as sp
from scipy.integrate import quad


# Function to calculate moment given initial conditions & interpolating functions
def moment_integral(temp_raw, x_vel, y_vel, X0, Y0, THETA0, K):
    # Set integration constants and such
    G = 2
    JET_E = 10 ** 9  # Jet energy in GeV
    V0 = 1

    # Parameterize path in terms of t from initial conditions
    def x_pos(t):
        return X0 + (V0 * np.cos(THETA0) * t)

    def y_pos(t):
        return Y0 + (V0 * np.cos(THETA0) * t)

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
        else:
            return True

    def temp_cut(t):
        if temp(t) < 0:
            return False
        else:
            return True

    # Determine final time for integral
    t_naut = np.amin(temp_raw.grid[0])
    t_final = np.amax(temp_raw.grid[0])

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
            return [3 * np.log(JET_E/mu(t)),0]  # No idea what the error should be here
        elif K == -4:
            i_integrand = lambda x : (x ** (1 + (K / 2))) * ((3 * x + 2) / ((x + 1) ** 3))
            return quad(i_integrand, 0, np.inf)[0]
        else:
            i_integrand = lambda x: (x ** (1 + (K / 2))) * ((3 * x + 2) / ((x + 1) ** 3))
            return quad(i_integrand, 0, np.inf)[0]

    # Define integrand - ONLY CORRECT FOR K=0 !!!!!!!!!!!
    def integrand(t):
        return (i_integral(t)[0])*(u_perp(t) / (1 - u_par(t))) * (mu(t) ** (K + 2)) * rho(t) * sigma(t)

    # Calculate moment point
    print('Evaluating moment integral...')
    raw_quad = sp.integrate.quad(integrand, t_naut, t_final)
    # Tack constants on
    moment = (1 / JET_E)*raw_quad[0]
    # Error on moment
    moment_error = raw_quad[1] # Not including I(k) error

    return moment, moment_error


# Error on the integral I is something to consider. Maxes out around 5 x 10^-8.

def err_demo():
    for i in np.arange(-4, 0, 0.1):
        k = i
        i_integrand = lambda x: (x ** (1 + (k / 2))) * ((3 * x + 2) / ((x + 1) ** 3))
        i_int = quad(i_integrand, 0, np.inf)
        print('k = ' + str(k) + ': ' + str(i_int))