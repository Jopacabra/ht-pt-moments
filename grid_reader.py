import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# Function to load a hydro grid as output by OSU-Hydro
def load_grid_file(grid_file_name):
    # Load grid data from file
    print('Loading grid data from file ...')
    grid_data = pd.read_table('backgrounds/' + grid_file_name, header=None, delim_whitespace=True, dtype=np.float64,
                              names=['time', 'xpos', 'ypos', 'temp', 'xvel', 'yvel'])
    # Set grid parameters
    # Grid is always square. Number of lines of the same time is
    # the number of grid squares == grid_width**2.
    print('Calculating grid parameters ...')
    grid_width = int(
        np.sqrt(grid_data['time'].value_counts().to_numpy()[-1]))  # Will be data-source-dependent in the future.
    n_grid_spaces = grid_width ** 2  # n_grid_spaces is total number of grid spaces / bins. Assumes square grid.
    # Number of time steps is grid_data's time column divided by number of grid spaces.
    NT = int(len(grid_data['time']) / n_grid_spaces)
    # Spit the data out
    return grid_data, grid_width, NT


# Function to interpolate a grid file
# Returns interpolating callable function
def interpolate_temp_grid(grid_data, grid_width, NT):
    print('Interpolating temp grid data ...')

    # Cut temp data out and convert to numpy array #
    temp_data_cut = grid_data[['temp']]
    temp_data = pd.DataFrame(temp_data_cut).to_numpy()

    # Reshape this nonsense into an array with first index time, then x, then y.
    # You can get the temperature at the grid point (ix,iy) at timestep it as temp_data[it,ix,iy].
    temp_data = np.reshape(temp_data, [NT, grid_width, grid_width])
    # Plot contour data
    # plt.contour(temp_data[0,:,:])
    # plt.show()

    # We get domains of time and space sets
    # Use this to match up to NT x grid_width x grid_width array of temperatures
    tList = np.ndarray.flatten(pd.DataFrame(grid_data['time']).to_numpy())  # Lists of time points individually
    xList = np.ndarray.flatten(pd.DataFrame(grid_data['xpos']).to_numpy())  # Lists of space points individually

    # Domains of physical times and positions individually
    tSpace = np.linspace(np.amin(tList), np.amax(tList), NT)  # Domain of time values
    xSpace = np.linspace(np.amin(xList), np.amax(xList), grid_width)  # Domain of space values

    # Interpolate data!
    interp_temps = RegularGridInterpolator((tSpace, xSpace, xSpace), temp_data)

    return interp_temps


# Function to interpolate the x velocities from a grid file
# Returns interpolating callable function
def interpolate_x_vel_grid(grid_data, grid_width, NT):
    print('Interpolating x vel. grid data ...')
    # Cut x velocity data out and convert to numpy array #
    vel_x_data = pd.DataFrame(grid_data['xvel']).to_numpy()

    # Reshape this nonsense into an array with first index time, then x, then y.
    # You can get the x velocity at the grid point (ix,iy) at timestep it as vel_x_data[it,ix,iy].
    vel_x_data = np.reshape(vel_x_data, [NT, grid_width, grid_width])

    # We get domains of time and space sets
    # Use this to match up to NT x grid_width x grid_width array of temperatures
    tList = np.ndarray.flatten(pd.DataFrame(grid_data['time']).to_numpy())  # Lists of time points individually
    xList = np.ndarray.flatten(pd.DataFrame(grid_data['xpos']).to_numpy())  # Lists of space points individually

    # Domains of physical times and positions individually
    tSpace = np.linspace(np.amin(tList), np.amax(tList), NT)  # Domain of time values
    xSpace = np.linspace(np.amin(xList), np.amax(xList), grid_width)  # Domain of space values

    # Interpolate data!
    interp_vel_x = RegularGridInterpolator((tSpace, xSpace, xSpace), vel_x_data)

    return interp_vel_x


# Function to interpolate the x velocities from a grid file
# Returns interpolating callable function
def interpolate_y_vel_grid(grid_data, grid_width, NT):
    print('Interpolating y vel. grid data ...')
    # Cut y vel data out and convert to numpy array #
    vel_y_data = pd.DataFrame(grid_data['yvel']).to_numpy()

    # Reshape this nonsense into an array with first index time, then x, then y.
    # You can get the y velocity at the grid point (ix,iy) at timestep it as vel_y_data[it,ix,iy].
    vel_y_data = np.reshape(vel_y_data, [NT, grid_width, grid_width])

    # We get domains of time and space sets
    # Use this to match up to NT x grid_width x grid_width array of temperatures
    tList = np.ndarray.flatten(pd.DataFrame(grid_data['time']).to_numpy())  # Lists of time points individually
    xList = np.ndarray.flatten(pd.DataFrame(grid_data['ypos']).to_numpy())  # Lists of space points individually

    # Domains of physical times and positions individually
    tSpace = np.linspace(np.amin(tList), np.amax(tList), NT)  # Domain of time values
    xSpace = np.linspace(np.amin(xList), np.amax(xList), grid_width)  # Domain of space values

    # Interpolate data!
    interp_vel_y = RegularGridInterpolator((tSpace, xSpace, xSpace), vel_y_data)

    return interp_vel_y


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


# Function to plot interpolated temperature function and / or velocity field
# Can plot contour or density / colormesh for temps, stream or quiver for velocities
# Takes the callable interpolated temperature function, x velocity function, and y velocity function. 
# Other options can adjust the output.
# Returns the plt.show() command to get the plot out.
def qgp_plot(temp_func, vel_x_func, vel_y_func, time, resolution=100, velresolution=20,
             temptype='contour', veltype='stream', plot_temp=True, plot_vel=True):
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
        temps = plt.pcolormesh(x_space, x_space, temp_points, cmap='plasma', shading='auto')
        plt.colorbar(temps)
    if temptype == 'contour' and plot_temp == True:
        temps = plt.contourf(x_space, x_space, temp_points, cmap='plasma')
        plt.colorbar(temps)
    if veltype == 'stream' and plot_vel == True:
        vels = plt.streamplot(vel_x_space, vel_x_space, x_vels, y_vels, color=np.sqrt(x_vels ** 2 + y_vels ** 2),
                              linewidth=1, cmap='rainbow')
        plt.colorbar(vels.lines)
    if veltype == 'quiver' and plot_vel == True:
        vels = plt.quiver(vel_x_space, vel_x_space, x_vels, y_vels, np.sqrt(x_vels ** 2 + y_vels ** 2), linewidth=1,
                          cmap='rainbow')
        plt.colorbar(vels)


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
