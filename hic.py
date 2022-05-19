import numpy as np
import scipy.stats as stats
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
# File will be created in directory "outputFile" and given name "0.dat".
def generateTrentoIC(bmin=None, bmax=None, projectile1='Au', projectile2='Au', outputFile=None, randomSeed=None,
                     normalization=None, crossSection=None, numEvents=1, quiet=False):
    # Create Trento command arguments
    # bmin and bmax control min and max impact parameter. Set to same value for specific b.
    # projectile1 and projectile2 control nucleon number and such for colliding nuclei.
    # numEvents determines how many to run and spit out a file for. Will be labeled like "0.dat", "1.dat" ...
    nucleon_width = 0.5  # In fm
    grid_step = .15*np.min([nucleon_width])  # In fm
    grid_max_target = 15  # In fm

    # Generate Trento command argument list
    trentoCmd = ['trento', '--number-events {}'.format(numEvents),
                 '--grid-step {} --grid-max {}'.format(grid_step, grid_max_target)]

    # Append any supplied commands in the proper order
    if projectile1 is not None:
        trentoCmd.append('--projectile {}'.format(projectile1))
    if projectile2 is not None:
        trentoCmd.append('--projectile {}'.format(projectile2))
    if bmin is not None:
        trentoCmd.append('--bmin {}'.format(bmin))  # Minimum impact parameter (in fm ???)
    if bmax is not None:
        trentoCmd.append('--bmax {}'.format(bmax))  # Maximum impact parameter (in fm ???)
    if outputFile is not None:
        trentoCmd.append('--output {}'.format(outputFile))  # Output file directory
    if randomSeed is not None:
        trentoCmd.append('--random-seed {}'.format(int(randomSeed)))  # Random seed for repeatability
    if normalization is not None:
        trentoCmd.append('--normalization {}'.format(normalization))  # Should be fixed by comp. to data multiplicity
    if crossSection is not None:
        trentoCmd.append('--cross-section {}'.format(crossSection))  # fm^2: http://qcd.phy.duke.edu/trento/usage.html

    # Run Trento command
    subprocess = utilities.run_cmd(*trentoCmd, quiet=quiet)  # Note star unpacks the list to pass the command list as arguments

    # Pass on result file name
    return outputFile, subprocess


# Function to generate a new trento IC for RHIC Kinematics:
# Au Au collisions at root-s of 200 GeV
# Normalization was fixed via multiplicity measures for 0-6% centrality guessed at via impact parameter
# Used center of 0-10% bin mult.:
# https://dspace.mit.edu/handle/1721.1/16933
# Nuclear cross section:
# https://inspirehep.net/literature/1394433
def generateRHICTrentoIC(bmin=None, bmax=None, outputFile=None, randomSeed=None, quiet=False):
    # Run Trento with known case parameters
    outputFile, subprocess = generateTrentoIC(bmin=bmin, bmax=bmax, projectile1='Au', projectile2='Au',
                                              outputFile=outputFile, randomSeed=randomSeed, normalization=7.6,
                                              crossSection=4.23, quiet=quiet)

    # Spit out the output
    return outputFile, subprocess


# Function to generate a new trento IC for LHC Kinematics:
# Pb Pb collisions at root-s of 5.02 TeV
# Normalization was fixed via multiplicity measures for 5% centrality
# (averaged 2.5-5% and 5-7.5%) guessed at via impact parameter
# https://arxiv.org/abs/1512.06104
# Nuclear cross section:
# https://inspirehep.net/literature/1190545
def generateLHCTrentoIC(bmin=None, bmax=None, outputFile=None, randomSeed=None, quiet=False):
    # Run Trento with known case parameters
    outputFile, subprocess = generateTrentoIC(bmin=bmin, bmax=bmax, projectile1='Pb', projectile2='Pb',
                                              outputFile=outputFile, randomSeed=randomSeed, normalization=19.5,
                                              crossSection=7.0, quiet=quiet)

    # Spit out the output
    return outputFile, subprocess


# Function that generates a load of samples for LHC conditions and finds centrality bins.
def centralityBoundsLHC(numSamples, bmin=None, bmax=None, percBinWidth=5, hist=False):
    multiplicityArray = np.array([])
    for i in range(0, numSamples):
        # Run Trento with given parameters
        subprocess = generateLHCTrentoIC(bmin=bmin, bmax=bmax, quiet=True)[1]

        # Parse output to list
        outputList = utilities.parseLine(subprocess.stdout)

        # Get and append multiplicity - the integrated reduced thickness function
        multiplicity = float(outputList[3])
        multiplicityArray = np.append(multiplicityArray, multiplicity)

    # Sort multiplicity array into ascending order
    multiplicityArray.sort()

    # Find how many samples should be in each bin
    binCap = int(utilities.round_decimals_down(multiplicityArray.size * (percBinWidth/100)))

    # Find bin edges
    i = 0
    binBounds = np.array([])
    while i < multiplicityArray.size:
        binBounds = np.append(binBounds, multiplicityArray[i])
        i += binCap

    # Tack on the max multiplicity, if it didn't get snagged.
    if binBounds[-1] != multiplicityArray[-1]:
        print('Last not snagged')
        binBounds = np.append(binBounds, multiplicityArray[-1])
    else:
        print('Last snagged')

    print('Bin bounds: ' + str(binBounds))

    if hist:
        # Calculate number of bins necessary using something like the Freedman-Diaconis rule
        # https://en.wikipedia.org/wiki/Freedman–Diaconis_rule
        binwidth = 2 * (stats.iqr(multiplicityArray)) / np.cbrt(multiplicityArray.size)
        numbins = int((np.amax(multiplicityArray) - np.amin(multiplicityArray)) / binwidth)
        # Create and show histogram
        plt.hist(multiplicityArray, bins=numbins)
        for bound in binBounds:
            plt.axvline(x=bound, color='black', ls=':', lw=1)
        plt.show()

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
