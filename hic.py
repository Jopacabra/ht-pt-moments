import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import math
import os
import logging
import utilities
import config
import freestream
import frzout

"""
This module is responsible for all processes related to jet production, event generation, & hard scattering.
- Rejection samples initial temperature profile for jet production
- Produces PDFs from temp backgrounds
- Planned inclusions: Pythia handling, etc.
"""

class StopEvent(Exception):
    """ Raise to end an event early. """

# Function that generates a new Trento collision event with parameters from config file.
# Returns the Trento output file name.
def runTrento(outputFile=False, randomSeed=None, numEvents=1, quiet=False, filename='initial.hdf'):
    # Make sure there's no file where we want to stick it.
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

    resultsDataFrame = pd.DataFrame(
        {
            "t_event": [],
            "b": [],
            "npart": [],
            "mult": [],
            "e2_re": [],
            "e2_im": [],
            "phi_2": [],
            "e3_re": [],
            "e3_im": [],
            "phi_3": [],
            "e4_re": [],
            "e4_im": [],
            "phi_4": [],
            "e5_re": [],
            "e5_im": [],
            "phi_5": [],
            "seed": [],
            "cmd": [],
        }
        )

    # Create Trento command arguments
    # bmin and bmax control min and max impact parameter. Set to same value for specific b.
    # projectile1 and projectile2 control nucleon number and such for colliding nuclei.
    # numEvents determines how many to run and spit out a file for. Will be labeled like "0.dat", "1.dat" ...

    # nucleon width default from DukeQCD was
    # nucleon_width = 0.5  # in fm
    # We have elected to keep this default.

    # grid_step detfault for DukeQCD was
    # grid_step = .15*np.min([nucleon_width])  # In fm
    # for better run-time and effectiveness, we've selected a default of 0.1 fm

    # Maximum grid size was by default 15 fm in DukeQCD. We have elected to keep this.
    # grid_max_target = 15  # In fm

    # Generate Trento command argument list
    trentoCmd = ['trento', '--number-events {}'.format(numEvents),
                 '--grid-step {} --grid-max {}'.format(config.transport.GRID_STEP, config.transport.GRID_MAX_TARGET)]

    # Append any supplied commands in the proper order
    trentoCmd.append('--projectile {}'.format(config.transport.trento.PROJ1))
    trentoCmd.append('--projectile {}'.format(config.transport.trento.PROJ2))
    if config.transport.trento.BMIN is not None:
        trentoCmd.append('--bmin {}'.format(config.transport.trento.BMIN))  # Minimum impact parameter (in fm ???)
    if config.transport.trento.BMAX is not None:
        trentoCmd.append('--bmax {}'.format(config.transport.trento.BMIN))  # Maximum impact parameter (in fm ???)
    if outputFile:
        trentoCmd.append('--output {}'.format(filename))  # Output file name
    if randomSeed is not None:
        trentoCmd.append('--random-seed {}'.format(int(randomSeed)))  # Random seed for repeatability
    trentoCmd.append('--normalization {}'.format(config.transport.trento.NORM))
    trentoCmd.append('--cross-section {}'.format(config.transport.trento.CROSS_SECTION))
    trentoCmd.append('--nucleon-width {}'.format(config.transport.trento.NUCLEON_WIDTH))


    # Run Trento command
    # Note star unpacks the list to pass the command list as arguments
    if not quiet:
        print('format: event_number impact_param npart mult e2_re e2_im e3_re e3_im e4_re e4_im e5_re e5_im')
    subprocess, output = utilities.run_cmd(*trentoCmd, quiet=quiet)

    # Parse output and pass to dataframe.
    for line in output:
        trentoOutput = line.split()
        trentoDataFrame = pd.DataFrame(
            {
                "t_event": [int(trentoOutput[0])],
                "b": [float(trentoOutput[1])],
                "npart": [float(trentoOutput[2])],
                "mult": [float(trentoOutput[3])],
                "e2_re": [float(trentoOutput[4])],
                "e2_im": [float(trentoOutput[5])],
                "phi_2": [float(trentoOutput[5])*(1/2)],
                "e3_re": [float(trentoOutput[6])],
                "e3_im": [float(trentoOutput[7])],
                "phi_3": [float(trentoOutput[7])*(1/3)],
                "e4_re": [float(trentoOutput[8])],
                "e4_im": [float(trentoOutput[9])],
                "phi_4": [float(trentoOutput[9])*(1/4)],
                "e5_re": [float(trentoOutput[10])],
                "e5_im": [float(trentoOutput[11])],
                "phi_5": [float(trentoOutput[11])*(1/5)],
                "seed": [randomSeed],
                "cmd": [trentoCmd],
            }
        )

        resultsDataFrame = resultsDataFrame.append(trentoDataFrame)

    # Pass on result file name, trentoSubprocess data, and dataframe.
    return resultsDataFrame.drop(labels='event', axis=1), filename, subprocess


# Function that generates a new Trento collision event with given parameters.
# Returns the Trento output file name.
def runTrentoLone(bmin=None, bmax=None, projectile1='Au', projectile2='Au', outputFile=False, randomSeed=None,
              normalization=None, crossSection=None, numEvents=1, quiet=False, grid_step=0.1, grid_max_target=15,
              nucleon_width=0.5, filename='initial.hdf'):
    # Make sure there's no file where we want to stick it.
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass

    resultsDataFrame = pd.DataFrame(
        {
            "t_event": [],
            "b": [],
            "npart": [],
            "mult": [],
            "e2_re": [],
            "e2_im": [],
            "phi_2": [],
            "e3_re": [],
            "e3_im": [],
            "phi_3": [],
            "e4_re": [],
            "e4_im": [],
            "phi_4": [],
            "e5_re": [],
            "e5_im": [],
            "phi_5": [],
            "seed": [],
            "cmd": [],
        }
        )

    # Create Trento command arguments
    # bmin and bmax control min and max impact parameter. Set to same value for specific b.
    # projectile1 and projectile2 control nucleon number and such for colliding nuclei.
    # numEvents determines how many to run and spit out a file for. Will be labeled like "0.dat", "1.dat" ...

    # nucleon width default from DukeQCD was
    # nucleon_width = 0.5  # in fm
    # We have elected to keep this default.

    # grid_step detfault for DukeQCD was
    # grid_step = .15*np.min([nucleon_width])  # In fm
    # for better run-time and effectiveness, we've selected a default of 0.1 fm

    # Maximum grid size was by default 15 fm in DukeQCD. We have elected to keep this.
    # grid_max_target = 15  # In fm

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
    if outputFile:
        trentoCmd.append('--output {}'.format(filename))  # Output file name
    if randomSeed is not None:
        trentoCmd.append('--random-seed {}'.format(int(randomSeed)))  # Random seed for repeatability
    if normalization is not None:
        trentoCmd.append('--normalization {}'.format(normalization))  # Should be fixed by comp. to data multiplicity
    if crossSection is not None:
        trentoCmd.append('--cross-section {}'.format(crossSection))  # fm^2: http://qcd.phy.duke.edu/trento/usage.html

    trentoCmd.append('--nucleon-width {}'.format(nucleon_width))


    # Run Trento command
    # Note star unpacks the list to pass the command list as arguments
    if not quiet:
        print('format: event_number impact_param npart mult e2_re e2_im e3_re e3_im e4_re e4_im e5_re e5_im')
    subprocess, output = utilities.run_cmd(*trentoCmd, quiet=quiet)

    # Parse output and pass to dataframe.
    for line in output:
        trentoOutput = line.split()
        trentoDataFrame = pd.DataFrame(
            {
                "t_event": [int(trentoOutput[0])],
                "b": [float(trentoOutput[1])],
                "npart": [float(trentoOutput[2])],
                "mult": [float(trentoOutput[3])],
                "e2_re": [float(trentoOutput[4])],
                "e2_im": [float(trentoOutput[5])],
                "phi_2": [float(trentoOutput[5])*(1/2)],
                "e3_re": [float(trentoOutput[6])],
                "e3_im": [float(trentoOutput[7])],
                "phi_3": [float(trentoOutput[7])*(1/3)],
                "e4_re": [float(trentoOutput[8])],
                "e4_im": [float(trentoOutput[9])],
                "phi_4": [float(trentoOutput[9])*(1/4)],
                "e5_re": [float(trentoOutput[10])],
                "e5_im": [float(trentoOutput[11])],
                "phi_5": [float(trentoOutput[11])*(1/5)],
                "seed": [randomSeed],
                "cmd": [trentoCmd],
            }
        )

        resultsDataFrame = resultsDataFrame.append(trentoDataFrame)

    # Pass on result file name, trentoSubprocess data, and dataframe.
    return resultsDataFrame.drop(labels='event', axis=1), filename, subprocess



# Define function to generate initial conditions object as for freestream input from trento file
def toFsIc(initial_file='initial.hdf', quiet=False):
    if not quiet:
        print('Packaging initial conditions array for: {}'.format(initial_file))

    with h5py.File(initial_file, 'r') as f:
        for dset in f.values():
            print(dset)
            ic = np.array(dset)
    return ic

# Function adapted from DukeQCD to run osu-hydro from the freestreamed initial conditions yielded by freestream
# Result files SHOULD be placed in the active folder.
def run_hydro(fs, event_size, grid_step=0.1, tau_fs=0.5, coarse=False, hydro_args=None, quiet=False,
              time_step=0.1, maxTime=None):
    """
    The handling of osu-hydro implemented here is adapted directly from DukeQCD's hic-eventgen package.
    https://github.com/Duke-QCD/hic-eventgen
    ---

    Run the initial condition contained in FreeStreamer object `fs` through
    osu-hydro on a grid with approximate physical size `event_size` [fm].
    Return a dict of freeze-out surface data suitable for passing directly
    to frzout.Surface.

    Initial condition arrays are cropped or padded as necessary.

    If `coarse` is an integer > 1, use only every `coarse`th cell from the
    initial condition arrays (thus increasing the physical grid step size
    by a factor of `coarse`).  Ignore the user input `hydro_args` and
    instead run ideal hydro down to a low temperature.

    `dt_ratio` sets the timestep as a fraction of the spatial step
    (dt = dt_ratio * dxy).  The SHASTA algorithm requires dt_ratio < 1/2.

    """
    dxy = grid_step * (coarse or 1)
    ls = math.ceil(event_size/dxy)  # the osu-hydro "ls" parameter
    n = 2*ls + 1  # actual number of grid cells

    for fmt, f, arglist in [
            ('ed', fs.energy_density, [()]),
            ('u{}', fs.flow_velocity, [(1,), (2,)]),
            ('pi{}{}', fs.shear_tensor, [(1, 1), (1, 2), (2, 2)]),
    ]:
        for a in arglist:
            X = f(*a)

            if coarse:
                X = X[::coarse, ::coarse]

            diff = X.shape[0] - n
            start = int(abs(diff)/2)

            if diff > 0:
                # original grid is larger -> cut out middle square
                s = slice(start, start + n)
                X = X[s, s]
            elif diff < 0:
                # original grid is smaller
                #  -> create new array and place original grid in middle
                Xn = np.zeros((n, n))
                s = slice(start, start + X.shape[0])
                Xn[s, s] = X
                X = Xn

            X.tofile(fmt.format(*a) + '.dat')

    dt = time_step

    if maxTime is not None:
        print('Limiting time...')
        hydroCmd = ['osu-hydro', 't0={} dt={} dxy={} nls={} maxt={}'.format(tau_fs, dt, dxy, ls, maxTime)]\
                   + hydro_args
    else:
        hydroCmd = ['osu-hydro', 't0={} dt={} dxy={} nls={}'.format(tau_fs, dt, dxy, ls)] + hydro_args

    hydroProc, hydroOutput = utilities.run_cmd(*hydroCmd, quiet=False)

    if not quiet:
        print('format: ITime, Time, Max Energy Density, Max Temp, iRegulateCounter, iRegulateCounterBulkPi')

    surface = np.fromfile('surface.dat', dtype='f8').reshape(-1, 16)

    # end event if the surface is empty -- this occurs in ultra-peripheral
    # events where the initial condition doesn't exceed Tswitch
    if surface.size == 0:
        raise StopEvent('empty surface')

    # surface columns:
    #   0    1  2  3         4         5         6    7
    #   tau  x  y  dsigma_t  dsigma_x  dsigma_y  v_x  v_y
    #   8     9     10    11    12    13    14    15
    #   pitt  pitx  pity  pixx  pixy  piyy  pizz  Pi

    # pack surface data into a dict suitable for passing to frzout.Surface
    return dict(
        zip(['x', 'sigma', 'v'], np.hsplit(surface, [3, 6, 8])),
        pi=dict(zip(['xx', 'xy', 'yy'], surface.T[11:14])),
        Pi=surface.T[15]
    )


# Function to generate a new HIC event and dump the files in the current working directory.
def generate_event(grid_max_target=config.transport.GRID_MAX_TARGET, grid_step=config.transport.GRID_STEP,
                   time_step=config.transport.TIME_STEP, tau_fs=config.transport.hydro.TAU_FS,
                   t_end=config.transport.hydro.T_END):
    logging.info('Generating new event.')

    # the "target" grid max: the grid shall be at least as large as the target
    # By defualt grid_max_target = config.transport.GRID_MAX_TARGET
    # next two lines set the number of grid cells and actual grid max,
    # which will be >= the target (same algorithm as trento)
    grid_n = math.ceil(2 * grid_max_target / grid_step)
    grid_max = .5 * grid_n * grid_step
    logging.info(
        'grid step = %.6f fm, n = %d, max = %.6f fm',
        grid_step, grid_n, grid_max
    )

    ##########
    # Trento #
    ##########

    # Choose random seed
    seed = int(np.random.uniform(0, 10000000000000000))
    logging.info('Random seed selected: {}'.format(seed))

    # Generate trento event
    trentoDataframe, trentoOutputFile, trentoSubprocess = runTrento(outputFile=True, randomSeed=seed,
                                                                    quiet=False,
                                                                    filename='initial.hdf')

    # Format trento data into initial conditions for freestream
    logging.info('Packaging trento initial conditions into array...')
    ic = toFsIc(initial_file='initial.hdf', quiet=False)

    #################
    # Freestreaming #
    #################
    # Freestream initial conditions
    logging.info('Freestreaming Trento conditions...')
    fs = freestream.FreeStreamer(initial=ic, grid_max=grid_max, time=tau_fs)

    # Important to close the hdf5 file.
    del ic

    #########
    # Hydro #
    #########
    # Run hydro on initial conditions
    # This is where we control the end point of the hydro. The HRG object created here has an energy density param.
    # that we use as the cut-off energy density for the hydro evolution. Doing things through frzout.HRG allows us to
    # specify a minimum temperature that will be enforced with the energy density popped out here.
    # create sampler HRG object (to be reused for all events)
    hrg_kwargs = dict(species='urqmd', res_width=True)
    hrg = frzout.HRG(t_end, **hrg_kwargs)

    # append switching energy density to hydro arguments
    eswitch = hrg.energy_density()
    hydro_args = ['edec={}'.format(eswitch)]

    # Coarse run to determine maximum radius
    logging.info('Running coarse hydro...')
    coarseHydroDict = run_hydro(fs, event_size=27, coarse=3, grid_step=grid_step,
                                tau_fs=tau_fs, hydro_args=hydro_args,
                                time_step=time_step)
    rmax = math.sqrt((
                             coarseHydroDict['x'][:, 1:3] ** 2
                     ).sum(axis=1).max())
    logging.info('rmax = %.3f fm', rmax)
    logging.info('rmax = {} fm'.format(rmax))

    # Determine maximum number of timesteps needed
    # This is the time it takes for a jet to travel across the plasma on its longest path at the speed of light
    maxTime = 2*rmax  # in fm --- equal to length to traverse in fm for c = 1 - 2x largest width of plasma
    logging.info('maxTime = %.3f fm', maxTime)
    logging.info('maxTime = {} fm'.format(maxTime))

    # Fine run
    logging.info('Running fine hydro...')
    run_hydro(fs, event_size=rmax, grid_step=grid_step, tau_fs=tau_fs,
              hydro_args=hydro_args, time_step=time_step, maxTime=maxTime)

    logging.info('Event generation complete')

    return trentoDataframe


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

    if plot:
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

    if plot:
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
