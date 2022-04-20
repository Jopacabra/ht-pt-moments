import os

import matplotlib.colors as colors
import numpy as np
import pandas as pd
import plasma
import plasma_interaction as pi
import hard_scattering as hs
import jets
# import ipympl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
hydro_file_path = askopenfilename(initialdir='/share/apps/Hybrid-Transport/hic-eventgen/results/')  # show an "Open" dialog box and return the path to the selected file
print('Selected file: ' + str(hydro_file_path))

"""
Select and load plasma grid file
"""

# Open grid file as object
hydro_file = plasma.osu_hydro_file(file_path=hydro_file_path)

# Interpolate temperatures and velocities
#interp_temp_func = hydro_file.interpolate_temp_grid()
#interp_x_vel_func = hydro_file.interpolate_x_vel_grid()
#interp_y_vel_func = hydro_file.interpolate_y_vel_grid()

# Create plasma object from current_event object
current_event = plasma.plasma_event(event=hydro_file)



# Find current_event parameters
tempMax = current_event.max_temp()
tempMin = current_event.max_temp()
t_naut = current_event.t0
t_final = current_event.tf

# Set moment to zero
moment = 0
angleDeflection = 0

# Select k moment
K = 0


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

# Define colorbar objects with "1" scalar mappable object.
tempcb = fig.colorbar(plt.cm.ScalarMappable(norm=None, cmap='rainbow'), ax=ax)
velcb = fig.colorbar(plt.cm.ScalarMappable(norm=None, cmap='rainbow'), ax=ax)


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
init_time = current_event.t0

# Make a horizontal slider to control the jet initial x position
axX0 = plt.axes([0.25, 0.04, 0.65, 0.03])
X0_slider = Slider(
    ax=axX0,
    label='X0 [fm]',
    valmin=current_event.xmin,
    valmax=current_event.xmax,
    valinit=init_X0,
)

# Make a vertically oriented slider to control the jet initial Y position
axY0 = plt.axes([0.05, 0.25, 0.0225, 0.63])
Y0_slider = Slider(
    ax=axY0,
    label="Y0 [fm]",
    valmin=current_event.ymin,
    valmax=current_event.ymax,
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
axEn = plt.axes([0.9, 0.25, 0.0225, 0.63])
En_slider = Slider(
    ax=axEn,
    label="Energy [GeV]",
    valmin=1,
    valmax=1000,
    valinit=100,
    orientation="vertical"
)

# Make a vertically oriented slider to control the jet energy
axTswitch = plt.axes([0.975, 0.25, 0.0225, 0.63])
tswitch_slider = Slider(
    ax=axTswitch,
    label="Temp Cutoff [GeV]",
    valmin=0,
    valmax=1,
    valinit=0,
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

# Create a `matplotlib.widgets.Button` to swap velocity plot type
axMoment = plt.axes([0, 0.08, 0.1, 0.02])
momentButton = Button(
    axMoment,
    'Moment',
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
    tswitch_slider.reset()


# Function to sample the plasma T^6 and set sliders to point
def sample_jet(event):
    sampledPoint = hs.generate_jet_point(current_event, 1)
    X0_slider.set_val(sampledPoint[0])
    Y0_slider.set_val(sampledPoint[1])


# Function to swap the global flag determining the velocity type
def swap_velType(event):
    global velType
    if velType == 'stream':
        velType = 'quiver'
    elif velType == 'quiver':
        velType = 'stream'
    update(0)


# Function to generate / update the plots
def update(val):
    global tempcb
    global velcb
    global current_jet

    # Clear the plots (without this, things will just stack)
    ax.clear()  # QGP plot
    for axisList in axs:  # Medium property plots
        for axis in axisList:
            axis.clear()

    # Clear the colorbars
    tempcb.remove()
    velcb.remove()

    # Set current_jet object to current slider parameters
    current_jet = jets.jet(x0=X0_slider.val, y0=Y0_slider.val,
                           theta0=THETA0_slider.val, event=current_event, energy=100)

    # Select QGP figure as current figure
    plt.figure(fig.number)
    # Select QGP plot axis as current axis
    plt.sca(ax)

    # Plot new temperatures & velocities
    tempPlot, velPlot, tempcb, velcb = current_event.plot(time_slider.val,
                                                          temp_resolution=100, vel_resolution=30, veltype=velType)


    timeRange = np.arange(t_naut, t_final, 0.1)
    t = np.array([])
    for time in timeRange:
        if pi.time_cut(current_event, time) and pi.pos_cut(current_event, current_jet, time) \
                and pi.temp_cut(current_event, current_jet, time):
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
    integrandArray = np.array([])

    # Calculate plot data
    for time in t:
        uPerp = current_event.u_perp(jet=current_jet, time=time)
        uPar = current_event.u_par(jet=current_jet, time=time)
        temp = current_event.temp(current_jet.coords3(time=time))
        vel = current_event.vel(jet=current_jet, time=time)
        overLambda = current_event.rho(jet=current_jet, time=time) * current_event.sigma(jet=current_jet, time=time)
        iInt = current_event.i_int_factor(jet=current_jet, time=time)
        xPOS = current_jet.xpos(time)
        yPOS = current_jet.ypos(time)
        integrand = pi.integrand(event=current_event, jet=current_jet, k=K, cutoffT=tswitch_slider.val)(time)

        uPerpArray = np.append(uPerpArray, uPerp)
        uParArray = np.append(uParArray, uPar)
        tempArray = np.append(tempArray, temp)
        velArray = np.append(velArray, vel)

        overLambdaArray = np.append(overLambdaArray, overLambda)

        integrandArray = np.append(integrandArray, integrand)

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
    axs[2, 3].plot(t, integrandArray)
    axs[2, 3].set_title("Integrand")

    # Plot vertical gridline at current time from slider
    for axisList in axs:  # Iterate through medium property plots
        for axis in axisList:
            axis.axvline(x=time_slider.val, color='black', ls=':')

    # Plot jet trajectory
    jetInitialX = current_jet.xpos(t[0])
    jetInitialY = current_jet.ypos(t[0])
    jetFinalX = current_jet.xpos(t[-1])
    jetFinalY = current_jet.ypos(t[-1])
    ax.plot([jetInitialX, jetFinalX], [jetInitialY, jetFinalY], ls=':', color='w')

    # Plot new jet position
    ax.plot(current_jet.xpos(time_slider.val), current_jet.ypos(time_slider.val), 'ro')

    # Refresh the canvas
    redraw(0)

# Set up update moment function

def calc_moment(val):
    global moment
    global angleDeflection

    moment = pi.moment_integral(current_event, current_jet, cutoffT=tswitch_slider.val)  # conversion factor fm from integration to GeV

    angleDeflection = np.arctan((moment[0] / current_jet.energy)) * (180 / np.pi)

    print("Jet produced at (" + str(X0_slider.val) + ", " + str(Y0_slider.val)
          + ") running at angle " + str(THETA0_slider.val) + ".")
    print("k=0 moment: " + str(moment[0]) + " GeV")
    print('Angular Deflection: ' + str(angleDeflection) + " deg.")


"""
Set up button and slider functions on change / click
"""

# Map Button Functions
resetButton.on_clicked(reset)
sampleButton.on_clicked(sample_jet)
updateButton.on_clicked(update)
velTypeButton.on_clicked(swap_velType)
momentButton.on_clicked(calc_moment)

# register the update function with each slider
time_slider.on_changed(update)
X0_slider.on_changed(update)
Y0_slider.on_changed(update)
THETA0_slider.on_changed(update)
En_slider.on_changed(update)
tswitch_slider.on_changed(update)



"""
Generate data and plot figures
"""
# Set current_jet object to current slider parameters
current_jet = jets.jet(x0=X0_slider.val, y0=Y0_slider.val,
                       theta0=THETA0_slider.val, event=current_event, energy=100)

# Draw all of the plots for the current (initial) slider positions
velType = 'stream'  # set default velocity plot type
update(0)

# Show the figures and wait for updates
plt.show()
