import numpy as np
import hic as hs


# Jet object class. Useful for plotting and simplifying everything.
# Note jet energy (energy) in GeV
# Note jet velocity (v0) in fraction of speed of light
# Note jet angle (theta0) in radians
class jet:
    # Instantiation statement. All parameters optional.
    def __init__(self, x0=None, y0=None, theta0=0, v0=1, t0=None, event=None, energy=100):
        # Set default values
        default_t0 = 0.48125  # Set default initial time for jet parametrization
        point = [0, 0]  # Set default jet production point

        # Initialize basic parameters
        self.theta0 = theta0
        self.v0 = v0
        self.energy = energy

        # If an event was provided for jet instantiation and either initial position was not defined,
        # try to sample the event for jet production. If this fails, set it to the default.
        # If nothing was provided to "event", just set to whatever x0 and y0 are (probably default)
        if not event is None and (x0 is None or y0 is None):
            try:
                point = hs.generate_jet_point(event.temp_func, num=1)
            except AttributeError:
                print("Jet event object not sample-able. Using default jet production point: " + str(point))
        if x0 is None:
            self.x0 = point[0]
        else:
            self.x0 = x0
        if y0 is None:
            self.y0 = point[1]
        else:
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
                print("Event object has no t0 and / or tf. Read failed. Jet t0 set to default, no tf.")

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
        if time is None:
            time = event.t0

        temp = event.temp_func(self.coords3(time=time))

        return temp

