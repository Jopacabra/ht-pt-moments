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
    def __init__(self, x0=None, y0=None, theta0=0, v0=1, t0=None, event=None, energy=100):
        logging.info('Creating new jet...')

        # Set default values
        default_t0 = 0.48125  # Set default initial time for jet parametrization
        point = [0, 0]  # Set default jet production point

        # Initialize basic parameters
        self.theta0 = theta0
        self.v0 = v0
        self.energy0 = energy
        self.energy = energy

        # Initialize shower correction, then sample shower correction distribution to determine post-shower direction
        self.shower_correction = 0
        self.shower_sample()

        # If an event was provided for jet instantiation and either initial position was not defined,
        # try to sample the event for jet production. If this fails, set it to the default.
        # If nothing was provided to "event", just set to whatever x0 and y0 are (probably default)
        if not event is None and (x0 is None or y0 is None):
            try:
                samPoint = hic.generate_jet_point(event)
                self.x0, self.y0 = samPoint[0], samPoint[1]
            except AttributeError:
                logging.warning("Jet event object not sample-able. Using default jet production point: " + str(point))
        elif event is None and (x0 is None or y0 is None):
            self.x0 = point[0]
            self.y0 = point[1]
        else:
            logging.info("Jet production point: {}, {}".format(x0, y0))
            self.x0 = x0
            self.y0 = y0


        # If an event was provided for jet instantiation, try to set jet t0 to event t0.
        # If this fails, set it to the default.
        # If nothing was provided to "event", just set to whatever t0 is (probably default)
        if event is None and t0 is None:
            self.t0 = default_t0
        elif not t0 is None:
            self.t0 = t0
        elif not event is None:
            try:
                self.t0 = event.t0
                self.tf = event.tf
            except (AttributeError, TypeError):
                self.t0 = default_t0
                logging.warning("Event object has no t0 and / or tf. Read failed. Jet t0 set to default, no tf.")

    # Method to obtain the current coordinates of the jet
    def xpos(self, time=None):
        if time is None:
            xpos = self.x0
        else:
            xpos = self.x0 + (self.v0 * np.cos(self.theta0) * (time - self.t0))
        return xpos

    # Method to obtain the current coordinates of the jet
    def ypos(self, time=None):
        if time is None:
            ypos = self.y0
        else:
            ypos = self.y0 + (self.v0 * np.sin(self.theta0) * (time - self.t0))
        return ypos

    # Method to obtain the current 2D coordinates of the jet
    def coords(self, time=None):
        if time is None:
            xpos = self.x0
            ypos = self.y0
        else:
            xpos = self.xpos(time)
            ypos = self.ypos(time)
        return np.array([xpos, ypos])

    # Method to obtain the current (2+1) coordinates of the jet
    def coords3(self, time=None):
        xpos = self.xpos(time)
        ypos = self.ypos(time)
        return np.array([time, xpos, ypos])

    # Method to obtain the current velocity of the jet
    # As of now, the velocity is always zero.
    def vel(self):
        return self.v0

    # Method to obtain the temperature in the given event at the jet's coordinates at given time.
    # As of now, the velocity is always zero.
    def temp(self, event, time=None):
        temperature = None
        if time is None:
            time = event.t0
        else:
            pass

        if pi.pos_cut(event=event, jet=self, time=time) and pi.pos_cut(event=event, jet=self, time=time):
            temperature = float(event.temp(self.coords3(time=time)))

        return temperature

    # Method to find the maximum time seen by the jet in the known background
    def max_temp(self, event):

        tempArray = np.array([])

        for t in np.arange(event.t0, event.tf, event.timestep):
            try:
                temperature = float(self.temp(event=event, time=t))
            except TypeError:
                temperature = 0
            tempArray = np.append(tempArray, temperature)

        maxTemp = np.amax(tempArray)
        return maxTemp

    # Method to sample a shower distribution and return a shower correction angle
    # As of now, this just returns zero
    def shower_sample(self):
        logging.info('Sampling shower correction distribution...')
        logging.debug('No shower correction for now!')
        self.shower_correction = 0


