import numpy as np
import logging
import hic
import plasma_interaction as pi

# Jet object class. Useful for plotting and simplifying everything.
# Note jet energy (energy) in GeV
# Note jet velocity (v0) in fraction of speed of light
# Note jet angle (theta0) in radians
import plasma_interaction


class jet:
    # Instantiation statement. All parameters optional.
    def __init__(self, x_0=0, y_0=0, phi_0=0, v_0=1, p_T0=100):
        logging.info('Creating new jet...')

        # Initialize basic parameters
        self.phi_0 = phi_0
        self.v_0 = v_0
        self.p_T0 = p_T0
        self.x_0 = x_0
        self.y_0 = y_0
        self.m = 0  # Jet is massless quark for now

        # Muck about with your coordinates
        self.p_x = self.p_T0 * np.cos(self.phi_0)
        self.p_y = self.p_T0 * np.sin(self.phi_0)

        # Initialize shower correction, then sample shower correction distribution to determine post-shower direction
        self.shower_correction = 0
        self.shower_sample()

        # Set current position and momentum values
        self.x = self.x_0
        self.y = self.y_0

    # Method to obtain the current 2D coordinates of the jet
    def coords(self):
        return np.array([self.x, self.y])

    # Method to obtain the current (2+1) coordinates of the jet
    def coords3(self, time=None):
        return np.array([time, self.x, self.y])

    # Method to obtain the current polar coordinates of the jet
    def polar_coords(self):
        phi = np.mod(np.arctan2(self.p_x, self.p_y), 2 * np.pi)
        rho = np.sqrt(self.x ** 2 + self.y ** 2)
        return np.array([rho, phi])

    # Method to sample a shower distribution and return a shower correction angle
    # As of now, this just returns zero
    def shower_sample(self):
        logging.info('Sampling shower correction distribution...')
        logging.debug('No shower correction for now!')
        self.shower_correction = 0

    # Method to add a given momentum in xy coordinates to the jet
    def add_q(self, dp_x=0, dp_y=0):
        self.p_x = self.p_x + dp_x
        self.p_y = self.p_y + dp_y

    # Method to add given relative perpendicular momentum to the jet
    def add_q_perp(self, q_perp):
        angle = self.polar_coords()[1] + np.pi / 2
        dp_x = q_perp * np.cos(angle)
        dp_y = q_perp * np.sin(angle)
        self.add_q(dp_x=dp_x, dp_y=dp_y)

    # Method to add given relative parallel momentum to the jet
    def add_q_par(self, q_par):
        angle = self.polar_coords()[1]
        dp_x = q_par * np.cos(angle)
        dp_y = q_par * np.sin(angle)
        self.add_q(dp_x=dp_x, dp_y=dp_y)

    # Method to propagate the jet for time tau
    def prop(self, tau=0):
        rho, phi = self.polar_coords()
        self.x = self.x + self.v_0 * np.cos(phi) * tau
        self.y = self.y + self.v_0 * np.sin(phi) * tau

    # Method to obtain the coordinates of the jet in tau amount of time
    def coords_in(self, tau=0):
        rho, phi = self.polar_coords()
        new_x = self.x + self.v_0 * np.cos(phi) * tau
        new_y = self.y + self.v_0 * np.sin(phi) * tau
        return np.array([new_x, new_y])

    # Method to obtain the coordinates of the jet in tau amount of time
    # given the current trajectory
    def coords3_in(self, tau=0, time=0):
        rho, phi = self.polar_coords()
        new_x = self.x + self.v_0 * np.cos(phi) * tau
        new_y = self.y + self.v_0 * np.sin(phi) * tau
        return np.array([time, new_x, new_y])

    # Method to obtain jet p_T
    def p_T(self):
        return np.sqrt(self.p_x**2 + self.p_y**2)


