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

###############
# Load events #
###############

# Open grid file
grid_data, grid_width, NT = pgr.load_grid_file('AuAu_Event_10.dat')

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

###############################
# Defining plasma plot object #
###############################

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



# Define initial parameters
init_amplitude = 5
init_frequency = 3
init_X0 = 0
init_Y0 = 0
init_THETA0 = 0
init_time = 0.5

############
# QGP Plot #
############

# Create the QGP Plot that will dynamically update
fig, ax = plt.subplots()

# Calculate & plot initial temps & velocities
temps = qgp_temps(temp_func, init_time, resolution=100)
tempPlot = ax.contourf(temps[0], temps[1], temps[2], cmap='plasma', vmin=tempMin, vmax=tempMax)
vels = qgp_vels(vel_x_func, vel_y_func, init_time, velresolution=20)
velPlot = ax.streamplot(vels[0], vels[1], vels[2], vels[3], color=np.sqrt(vels[2] ** 2 + vels[3] ** 2),
                              linewidth=1, cmap='rainbow', norm=colors.Normalize(vmin=0, vmax=1))
plt.colorbar(tempPlot)
plt.colorbar(velPlot.lines)

# Set plot labels
ax.set_xlabel('X Position [fm]')
ax.set_ylabel('Y Position [fm]')

# Plot initial jet position
ax.plot(init_X0, init_Y0, 'ro')

# adjust the main QGP plot to make room for the sliders
plt.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the jet initial x position
axX0 = plt.axes([0.25, 0.1, 0.65, 0.03])
X0_slider = Slider(
    ax=axX0,
    label='X0 [fm]',
    valmin=-5,
    valmax=5,
    valinit=init_X0,
)

# Make a vertically oriented slider to control the jet initial Y position
axY0 = plt.axes([0.1, 0.25, 0.0225, 0.63])
Y0_slider = Slider(
    ax=axY0,
    label="Y0 [fm]",
    valmin=-5,
    valmax=5,
    valinit=init_Y0,
    orientation="vertical"
)

# Make a horizontal slider to control the initial jet angle.
axTHETA0 = plt.axes([0.25, 0.9, 0.65, 0.03])
THETA0_slider = Slider(
    ax=axTHETA0,
    label='THETA0 [rad]',
    valmin=0,
    valmax=2*np.pi,
    valinit=init_THETA0,
)

# Make a horizontal slider to control the time.
axTime = plt.axes([0.25, 0.05, 0.65, 0.03])
time_slider = Slider(
    ax=axTime,
    label='Time [fm]',
    valmin=pi.t_naut(temp_func),
    valmax=pi.t_final(temp_func),
    valinit=init_time,
)

##########################
# Medium properties plot #
##########################

# Create the jet-medium plots that will dynamically update
fig1, axs = plt.subplots(3,4)
fig1.tight_layout() # Set plots not to overlap

# Set time as range from initial to final interaction times
t_final = pi.t_final(temp_func)
timeRange = np.arange(t_naut, t_final, 0.1)
t = np.array([])
for time in timeRange:
    if pi.time_cut(temp_func, time) and pi.pos_cut(temp_func, time, init_X0, init_Y0, init_THETA0, V0=1) and pi.temp_cut(temp_func,
                                                                                                          time, init_X0, init_Y0,
                                                                                                          init_THETA0, V0=1,
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
    uPerp = pi.u_perp(temp_func, vel_x_func, vel_y_func, time, init_X0, init_Y0, init_THETA0)
    uPar = pi.u_par(temp_func, vel_x_func, vel_y_func, time, init_X0, init_Y0, init_THETA0)
    temp = pi.temp(temp_func, time, init_X0, init_Y0, init_THETA0)
    vel = pi.vel_mag(temp_func, vel_x_func, vel_y_func, time, init_X0, init_Y0, init_THETA0)
    overLambda = pi.rho(temp_func, time, init_X0, init_Y0, init_THETA0) * pi.sigma(temp_func, time, init_X0, init_Y0, init_THETA0)
    iInt = pi.i_int_factor(temp_func, time, init_X0, init_Y0, init_THETA0, V0=1, JET_E=10)
    xPOS = pi.x_pos(time, init_X0, init_THETA0, V0=1, t_naut=0.5)
    yPOS = pi.y_pos(time, init_Y0, init_THETA0, V0=1, t_naut=0.5)

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









# The function to be called anytime a slider's value changes
def update(val):
    # Clear the plots (without this, things will just stack)
    ax.clear()

    # Calculate new temperatures & velocities
    newTemps = qgp_temps(temp_func, time_slider.val, resolution=100)
    newVels = qgp_vels(vel_x_func, vel_y_func, time_slider.val, velresolution=20)

    # Plot new temperatures & velocities
    ax.contourf(newTemps[0], newTemps[1], newTemps[2], cmap='plasma', vmin=tempMin, vmax=tempMax)
    ax.streamplot(newVels[0], newVels[1], newVels[2], newVels[3], color=np.sqrt(newVels[2] ** 2 + newVels[3] ** 2),
                            linewidth=1, cmap='rainbow', norm=colors.Normalize(vmin=0, vmax=1))

    # Plot new jet position
    ax.plot(pi.x_pos(time_slider.val, X0_slider.val, THETA0_slider.val, t_naut=t_naut), pi.y_pos(time_slider.val, Y0_slider.val, THETA0_slider.val, t_naut=t_naut), 'ro')

    # Refresh the canvas
    fig.canvas.draw_idle()

# Function to update medium properties
def update_medium(val):
    # Clear the plots (without this, things will just stack)
    axs[0, 0].clear()
    axs[0, 0].clear()
    axs[0, 1].clear()
    axs[0, 1].clear()
    axs[1, 0].clear()
    axs[1, 0].clear()
    axs[1, 1].clear()
    axs[1, 1].clear()
    axs[2, 0].clear()
    axs[2, 0].clear()
    axs[2, 1].clear()
    axs[2, 1].clear()
    axs[1, 2].clear()
    axs[1, 2].clear()
    axs[2, 2].clear()
    axs[2, 2].clear()
    axs[0, 2].clear()
    axs[0, 2].clear()
    axs[0, 3].clear()
    axs[0, 3].clear()
    axs[1, 3].clear()
    axs[1, 3].clear()

    timeRange = np.arange(t_naut, t_final, 0.1)
    t = np.array([])
    for time in timeRange:
        if pi.time_cut(temp_func, time) and pi.pos_cut(temp_func, time, X0_slider.val, Y0_slider.val, THETA0_slider.val,
                                                       V0=1) and pi.temp_cut(temp_func,
                                                                             time, X0_slider.val, Y0_slider.val, THETA0_slider.val, V0=1,
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
        overLambda = pi.rho(temp_func, time, X0_slider.val, Y0_slider.val, THETA0_slider.val) * pi.sigma(temp_func, time, X0_slider.val, Y0_slider.val, THETA0_slider.val)
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

def update_both(val):
    update(val)
    update_medium(val)

# register the update function with each slider
time_slider.on_changed(update)
X0_slider.on_changed(update_both)
Y0_slider.on_changed(update_both)
THETA0_slider.on_changed(update_both)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
resetButton = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    X0_slider.reset()
    Y0_slider.reset()
    THETA0_slider.reset()
    time_slider.reset()


resetButton.on_clicked(reset)


# Create a `matplotlib.widgets.Button` to randomly sample a jet position
sampleAx = plt.axes([0.9, 0.025, 0.1, 0.04])
sampleButton = Button(sampleAx, 'Sample Jet', hovercolor='0.975')


def sample_jet(event):
    sampledPoint = hs.generate_jet_point(temp_func, 1)
    X0_slider.set_val(sampledPoint[0])
    Y0_slider.set_val(sampledPoint[1])


sampleButton.on_clicked(sample_jet)

plt.show()