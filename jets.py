import numpy as np
import logging

# Jet object class. Useful for plotting and simplifying everything.
# Note jet energy (energy) in GeV
# Note jet velocity (v0) in fraction of speed of light
# Note jet angle (theta0) in radians



class jet:
    # Instantiation statement. All parameters optional.
    def __init__(self, x_0=0, y_0=0, phi_0=0, p_T0=100, part=None, tag=None, no=None, weight=1):
        logging.info('Creating new jet...')

        # Initialize basic parameters
        self.phi_0 = phi_0
        self.p_T0 = p_T0  # in GeV
        self.x_0 = x_0
        self.y_0 = y_0
        self.part = part
        self.weight = weight

        # If no particle type supplied, particle is a gluon (most common case).
        if self.part is None:
            self.part = 'g'

        # Quark masses - https://pdglive.lbl.gov/Particle.action?node=Q123&home=
        # Masses should be in GeV
        if self.part == 'u':
            self.m = 0.00216
        elif self.part == 'ubar':
            self.m = 0.00216  # anti-quark masses must be same!!!
        elif self.part == 'd':
            self.m = 0.00467
        elif self.part == 'dbar':
            self.m = 0.00467  # anti-quark masses must be same!!!
        elif self.part == 's':
            self.m = 0.0934
        elif self.part == 'sbar':
            self.m = 0.0934  # anti-quark masses must be same!!!
        elif self.part == 'g':
            self.m = 0  # Massless boson
        else:
            self.m = 0  # Default to massless gluon properties

        self.tag = tag
        self.no = no
        self.record = None

        # Muck about with your coordinates
        self.p_x = self.p_T0 * np.cos(self.phi_0)
        self.p_y = self.p_T0 * np.sin(self.phi_0)

        # Set calculated properties
        self.beta_0 = self.beta()

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
        phi = np.mod(np.arctan2(self.y, self.x), 2 * np.pi)
        rho = np.sqrt(self.x ** 2 + self.y ** 2)
        return np.array([rho, phi])

    # Method to obtain the current polar coordinates of the jet
    def polar_mom_coords(self):
        phi = np.mod(np.arctan2(self.p_y, self.p_x), 2 * np.pi)
        rho = np.sqrt(self.p_x ** 2 + self.p_y ** 2)
        return np.array([rho, phi])

    # Method to add a given momentum in xy coordinates to the jet
    def add_q(self, dp_x=0, dp_y=0):
        self.p_x = float(self.p_x + dp_x)
        self.p_y = float(self.p_y + dp_y)

    # Method to add given relative perpendicular momentum to the jet
    def add_q_perp(self, q_perp):
        angle = self.polar_mom_coords()[1] + (np.pi / 2)
        dp_x = q_perp * np.cos(angle)
        dp_y = q_perp * np.sin(angle)
        self.add_q(dp_x=dp_x, dp_y=dp_y)

    # Method to add given relative parallel momentum to the jet
    def add_q_par(self, q_par):
        angle = self.polar_mom_coords()[1]
        dp_x = q_par * np.cos(angle)
        dp_y = q_par * np.sin(angle)
        self.add_q(dp_x=dp_x, dp_y=dp_y)

    # Method to propagate the jet for time tau
    def prop(self, tau=0):
        rho, phi = self.polar_mom_coords()
        self.x = float(self.x + self.beta() * np.cos(phi) * tau)
        self.y = float(self.y + self.beta() * np.sin(phi) * tau)

    # Method to obtain the coordinates of the jet in tau amount of time
    def coords_in(self, tau=0):
        rho, phi = self.polar_mom_coords()
        new_x = self.x + self.beta() * np.cos(phi) * tau
        new_y = self.y + self.beta() * np.sin(phi) * tau
        return np.array([new_x, new_y])

    # Method to obtain the coordinates of the jet in tau amount of time
    # given the current trajectory
    def coords3_in(self, tau=0, time=0):
        rho, phi = self.polar_mom_coords()
        new_x = self.x + self.beta() * np.cos(phi) * tau
        new_y = self.y + self.beta() * np.sin(phi) * tau
        return np.array([time, new_x, new_y])

    # Method to obtain jet p_T
    def p_T(self):
        return np.sqrt(self.p_x**2 + self.p_y**2)

    # Method to obtain jet velocity (fraction of light speed)
    # Uses relativistic on-shell condition
    def beta(self):
        # On-shell condition with c=1: p = gamma * m * beta
        # beta is fraction of speed of light
        # beta = p / sqrt(m^2 + p^2)
        return self.p_T() / np.sqrt(self.m**2 + self.p_T()**2)


