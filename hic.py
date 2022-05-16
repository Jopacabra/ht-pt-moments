import numpy as np
import plasma as gr
import matplotlib.pyplot as plt
import utilities

"""
This module is responsible for all processes related to jet production, event generation, & hard scattering.
- Rejection samples initial temperature profile for jet production
- Produces PDFs from temp backgrounds
- Planned inclusions: Pythia handling, etc.
"""


# Function that generates a new Trento collision event with given parameters.
# Returns the Trento output file name.
# I THINK file will be created in current directory
def generateTrentoIC(bmin=None, bmax=None, projectile1='Au', projectile2='Au'):
    # Create Trento command arguments
    # bmin and bmax control min and max impact parameter. Set to same value for specific b.
    # projectile1 and projectile2 control nucleon number and such for colliding nuclei.
    numEvents = 1  # Number of collision ICs to generate
    nucleon_width = 0.5  # In fm
    grid_step = .15*np.min([nucleon_width])  # In fm
    grid_max_target = 15  # In fm
    trentoIC = 'trentoIC'  # Output file name

    if bmin is not None or bmax is not None:
        # Run Trento command
        subprocess = utilities.run_cmd(
            'trento',
            '--grid-step {} --grid-max {}'.format(grid_step, grid_max_target),
            '--output', trentoIC,
            '--nucleon-width {}'.format(nucleon_width),
            '--projectile {}'.format(projectile1),
            '--projectile {}'.format(projectile2),
            '--bmin {}'.format(bmin),
            '--bmax {}'.format(bmax)
        )
    else:
        # Run Trento command
        subprocess = utilities.run_cmd(
            'trento',
            '--grid-step {} --grid-max {}'.format(grid_step, grid_max_target),
            '--output', trentoIC,
            '--nucleon-width {}'.format(nucleon_width),
            '--projectile {}'.format(projectile1),
            '--projectile {}'.format(projectile2)
        )

    # Pass on result file name
    return trentoIC, subprocess


# Function that generates a new Trento collision event with given parameters.
# Returns the Trento output file name.
# I THINK file will be created in current directory
def centralityBounds(numSamples, bmin=None, bmax=None, projectile1='Au', projectile2='Au', percBinWidth=5):
    multiplicityArray = np.array([])
    for i in range(0, numSamples):
        # Run Trento with given parameters
        subprocess = generateTrentoIC(bmin=bmin, bmax=bmax, projectile1=projectile1, projectile2=projectile2)[1]

        # Parse output to list
        outputList = str(subprocess.stdout).split()

        # Get and append multiplicity
        multiplicity = outputList[4]
        multiplicityArray = np.append(multiplicityArray, multiplicity)

    # Sort multiplicity array into ascending order
    multiplicityArray.sort()

    # Find how many samples should be in each bin
    binCap = int(utilities.round_decimals_down(multiplicityArray.size() * (percBinWidth/100)))

    # Find bin edges
    i = 0
    binBounds = np.array([])
    while i < multiplicityArray.size():
        binBounds = np.append(binBounds, multiplicityArray[i])
        i += binCap

    # Tack on the max multiplicity, if it didn't get snagged.
    if binBounds[-1] != multiplicityArray[-1]:
        print('Last not snagged')
        binBounds = np.append(binBounds, multiplicityArray[-1])
    else:
        print('Last snagged')

    print('Bin bounds: ' + str(binBounds))

    # Pass on bin bounds
    return binBounds


# Function that defines a normalized 2D PDF array for a given interpolated temperature
# function's 0.5 fs (or given) timestep.
def jetprodPDF(temp_func, resolution=100, plot=False, initialTime=0.5):
    # Find spatial bounds of grid
    gridMin = np.amin(temp_func.grid[1])
    gridMax = np.amax(temp_func.grid[1])

    # Get initial timestep temperature grid with given resolution

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
    t_coords = np.full_like(x_coords, initialTime)

    # Put coordinates together into ordered pairs.
    points = np.transpose(np.array([t_coords, x_coords, y_coords]), (1, 2, 0))

    # Calculate temperatures
    initialGrid = temp_func(points)

    # Raise temps to 6th power
    raisedGrid = initialGrid ** 6

    # Rescale grid by adjusting values to between 0 and 1.
    minTemp = np.amin(raisedGrid)
    maxTemp = np.amax(raisedGrid)
    rescaledRaisedGrid = (raisedGrid - minTemp) / (maxTemp - minTemp)

    # Normalize the 2D array of initial temperatures
    normOfRaisedGrid = np.linalg.norm(raisedGrid, ord='nuc')
    normedRaisedGrid = raisedGrid / normOfRaisedGrid

    if plot == True:
        # Plot the normalized grid
        temps = plt.contourf(x_space, x_space, normedRaisedGrid, cmap='plasma')
        plt.colorbar(temps)
        plt.show()
    else:
        pass

    # return normedRaisedGrid
    return normedRaisedGrid


# Function that defines a normalized 2D PDF array for a given interpolated temperature
# function's 0.5 fs (or given) timestep.
def jetProdPDF_Function(temp_func, resolution=100, plot=False, initialTime=0.5):
    # Find spatial bounds of grid
    gridMin = np.amin(temp_func.grid[1])
    gridMax = np.amax(temp_func.grid[1])

    # Get initial timestep temperature grid with given resolution

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
    t_coords = np.full_like(x_coords, initialTime)

    # Put coordinates together into ordered pairs.
    points = np.transpose(np.array([t_coords, x_coords, y_coords]), (1, 2, 0))

    # Calculate temperatures
    initialGrid = temp_func(points)

    # Raise temps to 6th power
    raisedGrid = np.power(initialGrid, 6)
    Raised_Temp_Func = temp_func ** 6

    # Rescale grid by adjusting values to between 0 and 1.
    # minTemp = np.amin(raisedGrid)
    # maxTemp = np.amax(raisedGrid)
    # rescaledRaisedGrid = (raisedGrid - minTemp)/(maxTemp - minTemp)

    # Normalize the function of temperatures
    normOfRaisedGrid = np.linalg.norm(raisedGrid, ord='nuc')

    NormedRaised_Temp_Func = Raised_Temp_Func / normOfRaisedGrid

    if plot == True:
        pass
        # Plot the normalized grid
        # temps = plt.contourf(x_space, x_space, normedRaisedGrid, cmap='plasma')
        # plt.colorbar(temps)
        # plt.show()
    else:
        pass

    return NormedRaised_Temp_Func

# Generate a random (x, y, z) coordinate in a 3D box of l = w = boxSize and h = maxProb
# Origin at cent of bottom of box.
def cube_random(num=1, boxSize=1, maxProb=1):
    rng = np.random.default_rng()
    pointArray = np.array([])
    for i in np.arange(0, num):
        x = (boxSize * rng.random()) - (boxSize / 2)
        y = (boxSize * rng.random()) - (boxSize / 2)
        z = maxProb * rng.random()
        newPoint = np.array([x,y,z])
        if i == 0:
            pointArray = newPoint
        else:
            pointArray = np.vstack((pointArray, newPoint))
    return pointArray

# Function to rejection sample a given interpolated temperature function^6 for jet production.
# Returns an accepted (x, y) sample point as a numpy array.
def temp_6th_sample(event, maxAttempts=5, time='i', batch=1000):
    # Get temperature function
    temp_func = event.temp

    # Set time
    np.amin(temp_func.grid[0])
    if time == 'i':
        time = np.amin(temp_func.grid[0])
    elif time == 'f':
        time = np.amax(temp_func.grid[0])
    else:
        pass

    # Find max temp
    maxTemp = event.max_temp(time=time)

    # Find grid bounds
    gridMin = np.amin(temp_func.grid[1])
    gridMax = np.amax(temp_func.grid[1])
    gridWidth = gridMax - gridMin

    attempt = 0
    while attempt < maxAttempts:
        # Generate random point in 3D box of l = w = gridWidth and height maximum temp.^6
        # Origin at center of bottom of box
        pointArray = cube_random(num = batch, boxSize=gridWidth, maxProb=maxTemp**6)

        for point in pointArray:
            targetTemp = temp_func(np.array([time, point[0], point[1]]))**6

            # Check if point under 2D temp PDF curve
            if float(point[2]) < float(targetTemp):
                # If under curve, accept point and return
                # print("Attempt " + str(attempt) + " successful with point " + str(i) + "!!!")
                # print(point)
                # print("Random height: " + str(zPoints[i]))
                # print("Target <= height: " + str(float(targetTemp)))
                return point[0:2]
        print("Jet Production Sampling Attempt: " + str(attempt) + " failed.")
        attempt += 1
    print("Catastrophic error in jet production point sampling!")
    print("AHHHHHHHHHHHHHHH!!!!!!!!!!!")
    return "AHHHHHHHHHHHHHHH!!!!!!!!!!!"


# Function to generate a given number of jet production points
# sampled from the temperature^6 profile.
def generate_jet_point(event, num=1):
    pointArray = np.array([])
    for i in np.arange(0, num):
        newPoint = temp_6th_sample(event)
        if i == 0:
            pointArray = newPoint
        else:
            pointArray = np.vstack((pointArray, newPoint))
    return pointArray
