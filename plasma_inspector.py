import os

import matplotlib.colors as colors
import numpy as np
import pandas as pd
import plasma_grid_reader as pgr
import plasma_interaction as pi
import hard_scattering as hs
# import ipympl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
gridFilePath = askopenfilename(initialdir='/share/apps/Hybrid-Transport/hic-eventgen/results/')  # show an "Open" dialog box and return the path to the selected file
print('Selected file: ' + str(gridFilePath))

"""
Select and load plasma grid file
"""

# Open grid file
# grid_data, grid_width, NT = pgr.load_grid_file('AuAu_Event_10.dat')  # Hard code a file
grid_data, grid_width, NT = pgr.load_grid_file(gridFilePath, absolute=True)  # Select a file on run

# Interpolate temperatures
temp_func = pgr.interpolate_temp_grid(grid_data, grid_width, NT)

# Interpolate x velocities
vel_x_func = pgr.interpolate_x_vel_grid(grid_data, grid_width, NT)

# Interpolate y velocities
vel_y_func = pgr.interpolate_y_vel_grid(grid_data, grid_width, NT)

# Find event parameters
tempMax = pgr.max_temp(temp_func, resolution=100)
tempMin = 0
t_naut = pi.t_naut(temp_func)
t_final = pi.t_final(temp_func)

# Set moment to zero
moment = 0
angleDeflection = 0

"""
Define functions to calculate properties of the medium.
"""

# Function to return x space, y space, and temperatures to be fed into a contour plot.
def qgp_temps(temp_func, time, resolution=100):
    # Domains of physical positions to plot at (in fm)
    # These limits of the linear space obtain the largest and smallest input value for
    # the interpolating function's position inputs.
    grid_min = np.amin(temp_func.grid[1])
    grid_max = np.amax(temp_func.grid[1])
    x_space = np.linspace(grid_min, grid_max, resolution)
    y_space = x_space

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

    return x_space, y_space, temp_points, grid_min, grid_max



def qgp_vels(vel_x_func, vel_y_func, time, velresolution=20):
    # Domains of physical positions to plot at (in fm)
    # These limits of the linear space obtain the largest and smallest input value for
    # the interpolating function's position inputs.
    vel_x_space = np.linspace(np.amin(temp_func.grid[1]), np.amax(temp_func.grid[1]), velresolution)
    vel_y_space = vel_x_space

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

    return vel_x_space, vel_y_space, x_vels, y_vels

"""
Generate plot objects
"""

############
# QGP Plot #
############

# Create the QGP Plot that will dynamically update
fig, ax = plt.subplots()

# Set plot labels
ax.set_xlabel('X Position [fm]')
ax.set_ylabel('Y Position [fm]')

# adjust the main QGP plot to make room for the sliders
plt.subplots_adjust(left=0.1, bottom=0.1, top=0.9)

##########################
# Medium properties plot #
##########################

# Create the jet-medium plots that will dynamically update
fig1, axs = plt.subplots(3, 4, constrained_layout=True) # Constrained layout does some magic to organize the plots



"""
Create slider objects
"""
# Select QGP plot as current figure
plt.figure(fig.number)

# Set initial slider values
init_X0 = 0
init_Y0 = 0
init_THETA0 = 0
init_time = t_naut

# Make a horizontal slider to control the jet initial x position
axX0 = plt.axes([0.25, 0.04, 0.65, 0.03])
X0_slider = Slider(
    ax=axX0,
    label='X0 [fm]',
    valmin=-5,
    valmax=5,
    valinit=init_X0,
)

# Make a vertically oriented slider to control the jet initial Y position
axY0 = plt.axes([0.05, 0.25, 0.0225, 0.63])
Y0_slider = Slider(
    ax=axY0,
    label="Y0 [fm]",
    valmin=-5,
    valmax=5,
    valinit=init_Y0,
    orientation="vertical"
)

# Make a horizontal slider to control the initial jet angle.
axTHETA0 = plt.axes([0.25, 0.97, 0.65, 0.03])
THETA0_slider = Slider(
    ax=axTHETA0,
    label='THETA0 [rad]',
    valmin=0,
    valmax=2*np.pi,
    valinit=init_THETA0,
)

# Make a horizontal slider to control the time.
axTime = plt.axes([0.25, 0, 0.65, 0.03])
time_slider = Slider(
    ax=axTime,
    label='Time [fm]',
    valmin=t_naut,
    valmax=t_final,
    valinit=init_time,
)

# Make a vertically oriented slider to control the jet energy
axEn = plt.axes([0.95, 0.25, 0.0225, 0.63])
En_slider = Slider(
    ax=axEn,
    label="Energy [GeV]",
    valmin=1,
    valmax=1000,
    valinit=100,
    orientation="vertical"
)

"""
Create button objects
"""

# Select QGP plot as current figure
plt.figure(fig.number)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0, 0.06, 0.1, 0.02])
resetButton = Button(resetax, 'Reset', hovercolor='0.975')

# Create a `matplotlib.widgets.Button` to randomly sample a jet position
sampleAx = plt.axes([0, 0.04, 0.1, 0.02])
sampleButton = Button(sampleAx, 'Sample Jet', hovercolor='0.975')

# Create a `matplotlib.widgets.Button` to update everything by force.
updateax = plt.axes([0, 0.02, 0.1, 0.02])
updateButton = Button(updateax, 'Update', hovercolor='0.975')

# Create a `matplotlib.widgets.Button` to swap velocity plot type
axVelType = plt.axes([0, 0, 0.1, 0.02])
velTypeButton = Button(
    axVelType,
    'Vel Type',
    hovercolor='0.975'
)

"""
Functions for buttons and such
"""

# Function to redraw all figures
def redraw(event):
    fig.canvas.draw_idle()
    fig1.canvas.draw_idle()


# Function to reset sliders to default values
def reset(event):
    X0_slider.reset()
    Y0_slider.reset()
    THETA0_slider.reset()
    time_slider.reset()


# Function to sample the plasma T^6 and set sliders to point
def sample_jet(event):
    sampledPoint = hs.generate_jet_point(temp_func, 1)
    X0_slider.set_val(sampledPoint[0])
    Y0_slider.set_val(sampledPoint[1])


# Function to swap the global flag determining the velocity type
def swap_velType(event):
    global velType
    if velType == 'stream':
        velType = 'quiver'
    elif velType == 'quiver':
        velType = 'stream'
    redraw(event)


# Function to generate / update the plots
def update(val):
    # Clear the plots (without this, things will just stack)
    ax.clear()  # QGP plot
    for axisList in axs:  # Medium property plots
        for axis in axisList:
            axis.clear()

    # Calculate new temperatures & velocities
    newTemps = qgp_temps(temp_func, time_slider.val, resolution=100)
    newVels = qgp_vels(vel_x_func, vel_y_func, time_slider.val, velresolution=20)

    # Plot new temperatures & velocities
    tempPlot = ax.contourf(newTemps[0], newTemps[1], newTemps[2], cmap='plasma', vmin=tempMin, vmax=tempMax)
    # plt.colorbar(tempPlot, ax=ax)  # Currently stacking...

    if velType == 'quiver':
        velPlot = ax.quiver(newVels[0], newVels[1], newVels[2], newVels[3], np.sqrt(newVels[2] ** 2 + newVels[3] ** 2),
                            linewidth=1, cmap='rainbow', norm=colors.Normalize(vmin=0, vmax=1))
        # plt.colorbar(velPlot, ax=ax)  # Currently stacking...
    elif velType == 'stream':
        velPlot = ax.streamplot(newVels[0], newVels[1], newVels[2], newVels[3], color=np.sqrt(newVels[2] ** 2 + newVels[3] ** 2),
              linewidth=1, cmap='rainbow', norm=colors.Normalize(vmin=0, vmax=1))
        # plt.colorbar(velPlot.lines, ax=ax)  # Currently stacking...

    # Plot new jet position
    ax.plot(pi.x_pos(time_slider.val, X0_slider.val, THETA0_slider.val, t_naut=t_naut), pi.y_pos(time_slider.val, Y0_slider.val, THETA0_slider.val, t_naut=t_naut), 'ro')




    timeRange = np.arange(t_naut, t_final, 0.1)
    t = np.array([])
    for time in timeRange:
        if pi.time_cut(temp_func, time) and pi.pos_cut(temp_func, time, X0_slider.val, Y0_slider.val, THETA0_slider.val,
                                                       V0=1) and pi.temp_cut(temp_func,
                                                                             time, X0_slider.val, Y0_slider.val,
                                                                             THETA0_slider.val, V0=1,
                                                                             tempCutoff=0):
            t = np.append(t, time)
        else:
            break

    # Initialize empty arrays for the plot data
    uPerpArray = np.array([])
    uParArray = np.array([])
    tempArray = np.array([])
    velArray = np.array([])
    overLambdaArray = np.array([])
    iIntArray = np.array([])
    XArray = np.array([])
    YArray = np.array([])

    # Calculate plot data
    for time in t:
        uPerp = pi.u_perp(temp_func, vel_x_func, vel_y_func, time, X0_slider.val, Y0_slider.val, THETA0_slider.val)
        uPar = pi.u_par(temp_func, vel_x_func, vel_y_func, time, X0_slider.val, Y0_slider.val, THETA0_slider.val)
        temp = pi.temp(temp_func, time, X0_slider.val, Y0_slider.val, THETA0_slider.val)
        vel = pi.vel_mag(temp_func, vel_x_func, vel_y_func, time, X0_slider.val, Y0_slider.val, THETA0_slider.val)
        overLambda = pi.rho(temp_func, time, X0_slider.val, Y0_slider.val, THETA0_slider.val) * pi.sigma(temp_func,
                                                                                                         time,
                                                                                                         X0_slider.val,
                                                                                                         Y0_slider.val,
                                                                                                         THETA0_slider.val)
        iInt = pi.i_int_factor(temp_func, time, X0_slider.val, Y0_slider.val, THETA0_slider.val, V0=1, JET_E=10)
        xPOS = pi.x_pos(time, X0_slider.val, THETA0_slider.val, V0=1, t_naut=0.5)
        yPOS = pi.y_pos(time, Y0_slider.val, THETA0_slider.val, V0=1, t_naut=0.5)

        uPerpArray = np.append(uPerpArray, uPerp)
        uParArray = np.append(uParArray, uPar)
        tempArray = np.append(tempArray, temp)
        velArray = np.append(velArray, vel)

        overLambdaArray = np.append(overLambdaArray, overLambda)

        iIntArray = np.append(iIntArray, iInt)

        XArray = np.append(XArray, xPOS)
        YArray = np.append(YArray, yPOS)

    axs[0, 0].plot(t, uPerpArray)
    axs[0, 0].set_title("U_perp")
    axs[0, 1].plot(t, uParArray)
    axs[0, 1].set_title("U_par")
    axs[1, 0].plot(t, tempArray)
    axs[1, 0].set_title("Temp (GeV)")
    axs[1, 1].plot(t, velArray)
    axs[1, 1].set_title("Vel. Mag.")
    axs[2, 0].plot(t, (uPerpArray / (1 - uParArray)))
    axs[2, 0].set_title("perp / (1-par) (No Units)")
    axs[2, 1].plot(t, 1 / (5 * overLambdaArray))
    axs[2, 1].set_title("Lambda (fm)")
    axs[1, 2].plot(t, 1 / (overLambdaArray))
    axs[1, 2].set_title("Lambda (GeV^-1)")
    axs[2, 2].plot(t, 4 * tempArray ** 2)
    axs[2, 2].set_title("(gT)^2 ((GeV^2))")
    axs[0, 2].plot(t, iIntArray)
    axs[0, 2].set_title("I(k) Factor")
    axs[0, 3].plot(t, XArray)
    axs[0, 3].set_title("X Pos")
    axs[1, 3].plot(t, YArray)
    axs[1, 3].set_title("Y Pos")

    # Plot jet trajectory
    jetInitialX = pi.x_pos(t[0], X0_slider.val, THETA0_slider.val, V0=1, t_naut=t[0])
    jetInitialY = pi.y_pos(t[0], Y0_slider.val, THETA0_slider.val, V0=1, t_naut=t[0])
    jetFinalX = pi.x_pos(t[-1], X0_slider.val, THETA0_slider.val, V0=1, t_naut=t[0])
    jetFinalY = pi.y_pos(t[-1], Y0_slider.val, THETA0_slider.val, V0=1, t_naut=t[0])

    ax.plot([jetInitialX, jetFinalX], [jetInitialY, jetFinalY], ls=':', color='w')

    # Refresh the canvas
    redraw(0)

# Set up update moment function

def update_with_moment(val):
    global moment
    global angleDeflection
    global temp_func
    global vel_x_func
    global vel_y_func

    moment = pi.moment_integral(temp_func, vel_x_func,
                                vel_y_func, X0_slider.val, Y0_slider.val, THETA0_slider.val, 0,
                                JET_E=En_slider.val)  # conversion factor fm from integration to GeV

    angleDeflection = np.arctan((moment[0] / En_slider.val)) * (180 / np.pi)

    print("Jet produced at (" + str(X0_slider.val) + ", " + str(Y0_slider.val) + ") running at angle " + str(THETA0_slider.val) + ".")
    print("k=0 moment: " + str(moment[0]) + " GeV")
    print('Angular Deflection: ' + str(angleDeflection) + " deg.")


    update(val)


"""
Set up button and slider functions on change / click
"""

# Map Button Functions
resetButton.on_clicked(reset)
sampleButton.on_clicked(sample_jet)
updateButton.on_clicked(redraw)
velTypeButton.on_clicked(swap_velType)

# register the update function with each slider
time_slider.on_changed(update_with_moment)
X0_slider.on_changed(update_with_moment)
Y0_slider.on_changed(update_with_moment)
THETA0_slider.on_changed(update_with_moment)
En_slider.on_changed(update_with_moment)
velTypeButton.on_clicked(update)



"""
Generate data and plot figures
"""

# Draw all of the plots for the current (initial) slider positions
velType = 'stream'  # set default velocity plot type
update_with_moment(0)

# Show the figures and wait for updates
plt.show()
