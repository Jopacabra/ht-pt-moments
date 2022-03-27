import numpy as np
import pandas as pd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import quad
from config import G  # Coupling constant for strong interaction


class osu_hydro_file:
    def __init__(self, file_path, event_name=None):
        # Store your original file location and event number
        self.file_path = file_path
        self.name = event_name

        # Announce initialization
        print('Reading osu-hydro file ... event: ' + str(self.name))

        # Store grid data
        self.grid_data = pd.read_table(file_path, header=None, delim_whitespace=True, dtype=np.float64,
                              names=['time', 'xpos', 'ypos', 'temp', 'xvel', 'yvel'])

        # Initialize all the ordinary grid parameters

        # Grid is always square. Number of lines of the same time is the number of grid squares == grid_width**2.
        # Note that we check the END timestep because the first timestep may be repeated
        self.grid_width = int(np.sqrt(self.grid_data[['time']].value_counts().to_numpy()[-1]))

        # n_grid_spaces is total number of grid spaces / bins. Assumes square grid.
        self.n_grid_spaces = self.grid_width ** 2

        # Number of time steps is grid_data's time column divided by number of grid spaces.
        self.NT = int(len(self.grid_data[['time']]) / self.n_grid_spaces)

        # Set grid space & time domains & lists

        # We get lists of time and space coordinates from the file
        # These are literally just numpy arrays of the xpos and time columns
        # Use this to match up to NT x grid_width x grid_width array of values for interpolation
        self.xlist = np.ndarray.flatten(pd.DataFrame(self.grid_data[['xpos']]).to_numpy())
        self.tlist = np.ndarray.flatten(pd.DataFrame(self.grid_data[['time']]).to_numpy())

        # Difference in absolute time between steps in simulation
        # Note that we find the timestep from the end of the list. In files from osu-hydro,
        # the first two timesteps are labeled with the same absolute time.
        self.timestep = self.tlist[-1] - self.tlist[-1 - int(self.n_grid_spaces)]

        # Domains of physical times and positions
        """
        Absolute time for the first two steps from osu-hydro are the same.
        If we shift the first time step's absolute time back by the difference in timesteps,
        we can relabel the first step and keep the absolute times matching for every other step.
    
        If we do nothing, creating this linear time space will smoosh an extra timestep into the
        same initial and final time bounds. This, in principle, causes the evolution to happen 
        in the interpolated functions ever so slightly faster than it actually does in osu-hydro.
        This may be a small difference, but it is certainly non-physical.
    
        For now, we choose to shift back the first timestep by default.
        """
        # Domain of corrected time values
        self.tspace = np.linspace(np.amin(self.tlist) - self.timestep, np.amax(self.tlist), self.NT)
        # Domain of space values
        self.xspace = np.linspace(np.amin(self.xlist), np.amax(self.xlist), self.grid_width)

    # Method to get raw temp data
    def temp_array(self):
        # Cut temp data out and convert to numpy array
        temp_column = self.grid_data[['temp']]
        temp_data = pd.DataFrame(temp_column).to_numpy()

        # Reshape this nonsense into an array with first index time, then x, then y.
        # You can get the temperature at the grid indexes (ix,iy) at timestep 'it' as temp_array[it,ix,iy].
        temp_data = np.transpose(np.reshape(temp_data, [self.NT, self.grid_width, self.grid_width]), axes=[0, 2, 1])

        return temp_data

    # Method to get raw x velocity data
    def x_vel_array(self):
        # Cut x velocity data out and convert to numpy array
        x_vel_column = self.grid_data[['xvel']]
        x_vel_data = pd.DataFrame(x_vel_column).to_numpy()

        # Reshape this nonsense into an array with first index time, then x, then y.
        # You can get the x velocity at the grid point (ix,iy) at timestep it as vel_x_data[it,ix,iy].
        # The transposition has been confirmed against data.
        x_vel_data = np.transpose(np.reshape(x_vel_data, [self.NT, self.grid_width, self.grid_width]), axes=[0, 2, 1])

        return x_vel_data

    # Method to get raw y velocity data
    def y_vel_array(self):
        # Cut x velocity data out and convert to numpy array
        y_vel_column = self.grid_data[['yvel']]
        y_vel_data = pd.DataFrame(y_vel_column).to_numpy()

        # Reshape this nonsense into an array with first index time, then x, then y.
        # You can get the x velocity at the grid point (ix,iy) at timestep it as vel_x_data[it,ix,iy].
        # The transposition has been confirmed against data.
        y_vel_data = np.transpose(np.reshape(y_vel_data, [self.NT, self.grid_width, self.grid_width]), axes=[0, 2, 1])

        return y_vel_data

    # Method to plot raw temp data
    def plot_temps(self, time):
        # Cut temp data out and convert to numpy array #
        temp_data_cut = self.grid_data[['temp']]
        temp_data = pd.DataFrame(temp_data_cut).to_numpy()

        plt.contourf(temp_data[time, :, :])
        plt.show

    # Method to return interpolated function object from data
    # Function to interpolate a grid file
    # Returns interpolating callable function
    def interpolate_temp_grid(self):
        print('Interpolating temp grid data for event: ' + str(self.name))

        # Cut temp data out and convert to numpy array
        temp_data = self.temp_array()

        # Interpolate data!
        # The final temperatures have been confirmed directly against absolute coordinates in data.
        interp_temps = RegularGridInterpolator((self.tspace, self.xspace, self.xspace), temp_data)

        return interp_temps

    # Method to interpolate the x velocities from a grid file
    # Returns interpolating callable function
    def interpolate_x_vel_grid(self):
        print('Interpolating x vel. grid data for event: ' + str(self.name))

        # Cut x velocity data out and convert to numpy array
        x_vel_array = self.x_vel_array()

        # Interpolate data!
        # The final velocities have been confirmed directly against absolute coordinates in data.
        interp_x_vel = RegularGridInterpolator((self.tspace, self.xspace, self.xspace), x_vel_array)

        return interp_x_vel

    # Method to interpolate the y velocities from a grid file
    # Returns interpolating callable function
    def interpolate_y_vel_grid(self):
        print('Interpolating y vel. grid data for event: ' + str(self.name))

        # Cut y velocity data out and convert to numpy array
        y_vel_array = self.y_vel_array()

        # Interpolate data!
        # The final velocities have been confirmed directly against absolute coordinates in data.
        interp_y_vel = RegularGridInterpolator((self.tspace, self.xspace, self.xspace), y_vel_array)

        return interp_y_vel



# Plasma object as used for integration and muckery
class plasma_event:
    def __init__(self, temp_func=None, x_vel_func=None, y_vel_func=None, g=None):
        # Initialize all the ordinary plasma parameters
        self.temp = temp_func
        self.x_vel = x_vel_func
        self.y_vel = y_vel_func
        self.t0 = np.amin(self.temp.grid[0])
        self.tf = np.amax(self.temp.grid[0])
        self.xmin = np.amin(temp_func.grid[1])
        self.xmax = np.amax(temp_func.grid[1])
        self.ymin = np.amin(temp_func.grid[2])
        self.ymax = np.amax(temp_func.grid[2])

    # Method to return the total magnitude of the velocity at a given point
    def vel(self, point):
        return np.sqrt(self.x_vel(point)**2 + self.y_vel(point)**2)

    # Method to return angle of velocity vector at a given point
    def vel_angle(self, point):
        # np.arctan2 gives a signed angle, as opposed to np.arctan
        arctan2 = np.arctan2(self.y_vel(point), self.x_vel(point))

        # if the angle was negative, we need to correct it to return an angle on the domain [0, 2pi]
        if arctan2 < 0:
            return 2*np.pi + arctan2  # Here we add the negative angle, reducing to corresponding value on [0, 2pi]
        else:
            return arctan2

    # Method to return velocity perpendicular to given jet's trajectory at given time
    def u_perp(self, jet, time):
        return -self.x_vel(jet.coords3(time=time)) * np.sin(jet.theta0) \
               + self.y_vel(np.array(jet.coords3(time=time))) * np.cos(jet.theta0)

    # Method to return velocity parallel to given jet's trajectory at given time
    def u_par(self, jet, time):
        return self.x_vel(jet.coords3(time=time)) * np.cos(jet.theta0) \
               + self.y_vel(np.array(jet.coords3(time=time))) * np.sin(jet.theta0)

    # Method to return density at a particular point
    # Chosen to be ideal gluon gas dens. as per Sievert, Yoon, et. al.
    def rho(self, point=None, jet=None, time=None):
        # In point mode, give the density at the given point
        if jet is None and not point is None:
            current_point = point
        # In jet mode, return the density at the jet's position at given time
        elif not jet is None and point is None:
            current_point = jet.coords3(time=time)
        else:
            current_point = point
            print("Ill-defined density call")

        density = 1.202056903159594 * 16 * (1 / (np.pi ** 2)) * self.temp(current_point) ** 3

        return density

    # Method to return DeBye mass at a particular point
    # Chosen to be simple approximation. Ref - ???
    def mu(self, point=None, jet=None, time=None):
        # In point mode, give the DeBye mass at the given point
        if jet is None and not point is None:
            current_point = point
        # In jet mode, return the DeBye mass at the jet's position at given time
        elif not jet is None and point is None:
            current_point = jet.coords3(time=time)
        else:
            current_point = point
            print("Ill-defined DeBye mass call")

        debye_mass = G * self.temp(current_point)

        return debye_mass

    # Method to return total cross section at a particular point
    # Total GW cross section, as per Sievert, Yoon, et. al.
    def sigma(self, point=None, jet=None, time=None):
        """
        In the future, we can put in an if statement that determines if we're in a plasma state or hadron gas state.
        We can then return the appropriate cross section. This would require that this plasma object one day becomes
        simply an event object. This might make the object too heavy weight, but it would give us some very interesting
        powers.
        """
        # In point mode, give the cross section at the given point
        if jet is None and not point is None:
            current_point = point
        # In jet mode, return the cross section at the jet's position at given time
        elif not jet is None and point is None:
            current_point = jet.coords3(time=time)
        else:
            current_point = point
            print("Ill-defined cross section call")

        cross_section = np.pi * G ** 4 / (self.mu(point=current_point) ** 2)

        return cross_section

    def i_int_factor(self, point=None, jet=None, time=None, k=0, jetE=100):
        # In point mode, give the I(k) at the given point
        if jet is None and not point is None:
            current_point = point
            jetEnergy = jetE
        # In jet mode, return the I(k) at the jet's position at given time
        elif not jet is None and point is None:
            current_point = jet.coords3(time=time)
            jetEnergy = jet.energy
        else:
            jetEnergy = jet.energy
            current_point = point
            print("Ill-defined I(k) call")

        if k == 0:
            Ik = 3 * np.log(jetEnergy / self.mu(point=current_point))  # No idea what the error should be here
        else:  # Not really a thing.
            print('I(k) for k =/= 0 is not functional. Using k=0 form.')
            Ik = 3 * np.log(jetEnergy / self.mu(point=current_point))  # No idea what the error should be here

        return Ik


# Function to plot interpolated temperature function
def temp_plot(temp_func, time, resolution):
    # Domains of physical positions to plot at (in fm)
    # These limits of the linear space obtain the largest and smallest input value for
    # the interpolating function's position inputs.
    x_space = np.linspace(np.amin(temp_func.grid[1]), np.amax(temp_func.grid[1]), resolution)

    # Create arrays of each coordinate
    # E.g. Here x_coords is a 2D array showing the x coordinates of each cell
    # We necessarily must set time equal to a constant to plot in 2D.
    x_coords, y_coords = np.meshgrid(x_space, x_space, indexing='ij')
    # t_coords set to be an array matching the length of x_coords full of constant time
    t_coords = np.full_like(x_coords, time)

    # Put coordinates together into an ordered pair.
    points = np.transpose(np.array([t_coords, x_coords, y_coords]), (1, 2, 0))

    temp_points = temp_func(points)

    plt.contour(x_space, x_space, temp_points)
    return plt.show()


# Function to plot interpolated temperature function
def temp_plot_contour(temp_func, time, resolution):
    # Domains of physical positions to plot at (in fm)
    # These limits of the linear space obtain the largest and smallest input value for
    # the interpolating function's position inputs.
    x_space = np.linspace(np.amin(temp_func.grid[1]), np.amax(temp_func.grid[1]), resolution)

    # Create arrays of each coordinate
    # E.g. Here x_coords is a 2D array showing the x coordinates of each cell
    # We necessarily must set time equal to a constant to plot in 2D.
    x_coords, y_coords = np.meshgrid(x_space, x_space, indexing='ij')
    # t_coords set to be an array matching the length of x_coords full of constant time
    t_coords = np.full_like(x_coords, time)

    # Put coordinates together into an ordered pair.
    points = np.transpose(np.array([t_coords, x_coords, y_coords]), (1, 2, 0))

    temp_points = temp_func(points)

    plt.contour(x_space, x_space, temp_points, cmap='plasma')
    plt.colorbar()
    return plt.show()


# Function to plot interpolated temperature function
def temp_plot_density(temp_func, time, resolution):
    # Domains of physical positions to plot at (in fm)
    # These limits of the linear space obtain the largest and smallest input value for
    # the interpolating function's position inputs.
    x_space = np.linspace(np.amin(temp_func.grid[1]), np.amax(temp_func.grid[1]), resolution)

    # Create arrays of each coordinate
    # E.g. Here x_coords is a 2D array showing the x coordinates of each cell
    # We necessarily must set time equal to a constant to plot in 2D.
    x_coords, y_coords = np.meshgrid(x_space, x_space, indexing='ij')
    # t_coords set to be an array matching the length of x_coords full of constant time
    t_coords = np.full_like(x_coords, time)

    # Put coordinates together into an ordered pair.
    points = np.transpose(np.array([t_coords, x_coords, y_coords]), (1, 2, 0))

    temp_points = temp_func(points)

    plt.pcolormesh(x_space, x_space, temp_points, cmap='plasma', shading='auto')
    plt.colorbar()
    return plt.show()


# Function to plot interpolated velocity function
def vel_plot(vel_x_func, vel_y_func, time, resolution):
    # Domains of physical positions to plot at (in fm)
    # These limits of the linear space obtain the largest and smallest input value for
    # the interpolating function's position inputs.
    x_space = np.linspace(np.amin(vel_x_func.grid[1]), np.amax(vel_x_func.grid[1]), resolution)

    # Create arrays of each coordinate
    # E.g. Here x_coords is a 2D array showing the x coordinates of each cell
    # We necessarily must set time equal to a constant to plot in 2D.
    x_coords, y_coords = np.meshgrid(x_space, x_space, indexing='ij')
    # t_coords set to be an array matching the length of x_coords full of constant time
    t_coords = np.full_like(x_coords, time)

    # Put coordinates together into an ordered pair.
    points = np.transpose(np.array([t_coords, x_coords, y_coords]), (1, 2, 0))

    x_vels = vel_x_func(points)
    y_vels = vel_y_func(points)

    plt.quiver(x_space, x_space, x_vels, y_vels)
    return plt.show()


# Function to find the maximum temperature of a particular grid at the initial time or a particular time.
def max_temp(temp_func, resolution=100, time='i'):
    # Find max temp
    # Get initial timestep temperature grid with given resolution
    gridMin = np.amin(temp_func.grid[1])
    gridMax = np.amax(temp_func.grid[1])

    if time == 'i':
        time = np.amin(temp_func.grid[0])
    elif time == 'f':
        time = np.amax(temp_func.grid[0])
    else:
        pass

    # Adapted from grid_reader.qgp_plot()
    #
    # Domains of physical positions to plot at (in fm)
    # These limits of the linear space obtain the largest and smallest input value for
    # the interpolating function's position inputs.
    x_space = np.linspace(gridMin, gridMax, resolution)

    # Create arrays of each coordinate
    # E.g. Here x_coords is a 2D array showing the x coordinates of each cell
    # We necessarily must set time equal to a constant to plot in 2D.
    x_coords, y_coords = np.meshgrid(x_space, x_space, indexing='ij')
    # t_coords set to be an array matching the length of x_coords full of constant time
    # Note that we select "initial time" as 0.5 fs by default
    t_coords = np.full_like(x_coords, time)

    # Put coordinates together into ordered pairs.
    points = np.transpose(np.array([t_coords, x_coords, y_coords]), (1, 2, 0))

    # Calculate temperatures and take maximum.
    maxTemp = np.amax(temp_func(points))

    return maxTemp

# Function to find the minimum temperature of a particular grid at the initial time or a particular time.
def min_temp(temp_func, resolution=100, time='i'):
    # Get initial timestep temperature grid with given resolution
    gridMin = np.amin(temp_func.grid[1])
    gridMax = np.amax(temp_func.grid[1])

    if time == 'i':
        time = np.amin(temp_func.grid[0])
    elif time == 'f':
        time = np.amax(temp_func.grid[0])
    else:
        pass

    # Adapted from grid_reader.qgp_plot()
    #
    # Domains of physical positions to plot at (in fm)
    # These limits of the linear space obtain the largest and smallest input value for
    # the interpolating function's position inputs.
    x_space = np.linspace(gridMin, gridMax, resolution)

    # Create arrays of each coordinate
    # E.g. Here x_coords is a 2D array showing the x coordinates of each cell
    # We necessarily must set time equal to a constant to plot in 2D.
    x_coords, y_coords = np.meshgrid(x_space, x_space, indexing='ij')
    # t_coords set to be an array matching the length of x_coords full of constant time
    # Note that we select "initial time" as 0.5 fs by default
    t_coords = np.full_like(x_coords, time)

    # Put coordinates together into ordered pairs.
    points = np.transpose(np.array([t_coords, x_coords, y_coords]), (1, 2, 0))

    # Calculate temperatures and take maximum.
    minTemp = np.amin(temp_func(points))

    return minTemp

# Function to plot interpolated temperature function and / or velocity field
# Can plot contour or density / colormesh for temps, stream or quiver for velocities
# Takes the callable interpolated temperature function, x velocity function, and y velocity function. 
# Other options can adjust the output.
# Returns the plt.show() command to get the plot out.
def qgp_plot(temp_func, vel_x_func, vel_y_func, time, resolution=100, velresolution=20,
             temptype='contour', veltype='stream', plot_temp=True, plot_vel=True):
    tempMax = max_temp(temp_func)

    if plot_temp == True:
        # Domains of physical positions to plot at (in fm)
        # These limits of the linear space obtain the largest and smallest input value for
        # the interpolating function's position inputs.
        x_space = np.linspace(np.amin(temp_func.grid[1]), np.amax(temp_func.grid[1]), resolution)

        # Create arrays of each coordinate
        # E.g. Here x_coords is a 2D array showing the x coordinates of each cell
        # We necessarily must set time equal to a constant to plot in 2D.
        x_coords, y_coords = np.meshgrid(x_space, x_space, indexing='ij')
        # t_coords set to be an array matching the length of x_coords full of constant time
        t_coords = np.full_like(x_coords, time)

        # Put coordinates together into an ordered pair.
        points = np.transpose(np.array([t_coords, x_coords, y_coords]), (1, 2, 0))

        # Calculate temperatures
        temp_points = temp_func(points)

    if plot_vel == True:
        # Domains of physical positions to plot at (in fm)
        # These limits of the linear space obtain the largest and smallest input value for
        # the interpolating function's position inputs.
        vel_x_space = np.linspace(np.amin(temp_func.grid[1]), np.amax(temp_func.grid[1]), velresolution)

        # Create arrays of each coordinate
        # E.g. Here x_coords is a 2D array showing the x coordinates of each cell
        # We necessarily must set time equal to a constant to plot in 2D.
        vel_x_coords, vel_y_coords = np.meshgrid(vel_x_space, vel_x_space, indexing='ij')

        # t_coords set to be an array matching the length of x_coords full of constant time
        vel_t_coords = np.full_like(vel_x_coords, time)

        # t_coords set to be an array matching the length of x_coords full of constant time
        vel_points = np.transpose(np.array([vel_t_coords, vel_x_coords, vel_y_coords]), (1, 2, 0))

        # Calculate velocities
        x_vels = vel_x_func(vel_points)
        y_vels = vel_y_func(vel_points)

    # Make plots    
    if temptype == 'density' and plot_temp == True:
        temps = plt.pcolormesh(x_space, x_space, temp_points, cmap='plasma', shading='auto', vmin=0, vmax=tempMax)
        plt.colorbar(temps)
    if temptype == 'contour' and plot_temp == True:
        temps = plt.contourf(x_space, x_space, temp_points, cmap='plasma', vmin=0, vmax=tempMax)
        plt.colorbar(temps)
    if veltype == 'stream' and plot_vel == True:
        vels = plt.streamplot(vel_x_space, vel_x_space, x_vels, y_vels, color=np.sqrt(x_vels ** 2 + y_vels ** 2),
                              linewidth=1, cmap='rainbow', norm=colors.Normalize(vmin=0, vmax=1))
        plt.colorbar(vels.lines)
    if veltype == 'quiver' and plot_vel == True:
        vels = plt.quiver(vel_x_space, vel_x_space, x_vels, y_vels, np.sqrt(x_vels ** 2 + y_vels ** 2), linewidth=1,
                          cmap='rainbow', norm=colors.Normalize(vmin=0, vmax=1))
        plt.colorbar(vels)



    

