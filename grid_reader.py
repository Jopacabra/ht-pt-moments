import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


def load_grid_file(grid_file_name):
    # Load grid data from file
    grid_data = pd.read_fwf('backgrounds/' + grid_file_name, header=None,
                            names=['time', 'xpos', 'ypos', 'temp', 'xvel', 'yvel'])
    # Set grid parameters
    grid_width = 165  # Will be data-dependent in the future.
    n_grid_spaces = grid_width ** 2  # n_grid_spaces is total number of grid spaces / bins. Assumes square grid.
    # Number of time steps is grid_data's time column divided by number of grid spaces.
    NT = int(len(grid_data['time']) / n_grid_spaces)
    # Spit the data out
    return grid_data, grid_width, NT


# Function to interpolate a grid file
# Returns interpolating callable function
def interpolate_temp_grid(grid_data, grid_width, NT):


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
