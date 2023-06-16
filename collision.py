import numpy as np
import pandas as pd
from scipy import interpolate
try:
    import matplotlib.pyplot as plt
except:
    print('NO MATPLOTLIB')
import h5py
import math
import os
import logging
import utilities
import config
from hic import flow
from utilities import cube_random

try:
    import freestream
except:
    logging.warning('freestream not found.')
try:
    import frzout
except:
    logging.warning('frzout not found.')

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
            "event": [],
            "b": [],
            "npart": [],
            "mult": [],
            "e2_re": [],
            "e2_im": [],
            "psi_2": [],
            "e3_re": [],
            "e3_im": [],
            "psi_3": [],
            "e4_re": [],
            "e4_im": [],
            "psi_4": [],
            "e5_re": [],
            "e5_im": [],
            "psi_5": [],
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
        logging.info('format: event_number impact_param npart mult e2_re e2_im e3_re e3_im e4_re e4_im e5_re e5_im')
    subprocess, output = utilities.run_cmd(*trentoCmd, quiet=quiet)

    # Parse output and pass to dataframe.
    for line in output:
        trentoOutput = line.split()
        try:
            trentoDataFrame = pd.DataFrame(
                {
                    "event": [int(trentoOutput[0])],
                    "b": [float(trentoOutput[1])],
                    "npart": [float(trentoOutput[2])],
                    "mult": [float(trentoOutput[3])],
                    "e2_re": [float(trentoOutput[4])],
                    "e2_im": [float(trentoOutput[5])],
                    "psi_2": [float((1/2) * np.arctan2(float(trentoOutput[5]), float(trentoOutput[4])))],
                    "e3_re": [float(trentoOutput[6])],
                    "e3_im": [float(trentoOutput[7])],
                    "psi_3": [float((1/3) * np.arctan2(float(trentoOutput[7]), float(trentoOutput[6])))],
                    "e4_re": [float(trentoOutput[8])],
                    "e4_im": [float(trentoOutput[9])],
                    "psi_4": [float((1/4) * np.arctan2(float(trentoOutput[9]), float(trentoOutput[8])))],
                    "e5_re": [float(trentoOutput[10])],
                    "e5_im": [float(trentoOutput[11])],
                    "psi_5": [float((1/5) * np.arctan2(float(trentoOutput[11]), float(trentoOutput[10])))],
                    "seed": [randomSeed],
                    "cmd": [trentoCmd],
                }
            )
        except ValueError:
            pass

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
            "event": [],
            "b": [],
            "npart": [],
            "mult": [],
            "e2_re": [],
            "e2_im": [],
            "psi_2": [],
            "e3_re": [],
            "e3_im": [],
            "psi_3": [],
            "e4_re": [],
            "e4_im": [],
            "psi_4": [],
            "e5_re": [],
            "e5_im": [],
            "psi_5": [],
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
        logging.info('format: event_number impact_param npart mult e2_re e2_im e3_re e3_im e4_re e4_im e5_re e5_im')
    subprocess, output = utilities.run_cmd(*trentoCmd, quiet=quiet)

    # Parse output and pass to dataframe.
    for line in output:
        trentoOutput = line.split()
        try:
            trentoDataFrame = pd.DataFrame(
                {
                    "event": [int(trentoOutput[0])],
                    "b": [float(trentoOutput[1])],
                    "npart": [float(trentoOutput[2])],
                    "mult": [float(trentoOutput[3])],
                    "e2_re": [float(trentoOutput[4])],
                    "e2_im": [float(trentoOutput[5])],
                    "psi_2": [float((1 / 2) * np.arctan2(float(trentoOutput[5]), float(trentoOutput[4])))],
                    "e3_re": [float(trentoOutput[6])],
                    "e3_im": [float(trentoOutput[7])],
                    "psi_3": [float((1 / 3) * np.arctan2(float(trentoOutput[7]), float(trentoOutput[6])))],
                    "e4_re": [float(trentoOutput[8])],
                    "e4_im": [float(trentoOutput[9])],
                    "psi_4": [float((1 / 4) * np.arctan2(float(trentoOutput[9]), float(trentoOutput[8])))],
                    "e5_re": [float(trentoOutput[10])],
                    "e5_im": [float(trentoOutput[11])],
                    "psi_5": [float((1 / 5) * np.arctan2(float(trentoOutput[11]), float(trentoOutput[10])))],
                    "seed": [randomSeed],
                    "cmd": [trentoCmd],
                }
            )
        except ValueError:
            pass

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
                   t_end=config.transport.hydro.T_END, seed=None, get_rmax=False, working_dir=None):

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
    if seed is None:
        seed = int(np.random.uniform(0, 10000000000000000))
    logging.info('Random seed selected: {}'.format(seed))

    # Decide where to locate the initial conditions file
    if working_dir is not None:
        trento_ic_path = working_dir + '/initial.hdf'
    else:
        trento_ic_path = 'initial.hdf'

    # Debug pwd
    print('Running trento in...')
    print(os.getcwd())

    # Generate trento event
    event_dataframe, trento_output_file, trento_subprocess = runTrento(outputFile=True, randomSeed=seed,
                                                                    quiet=False,
                                                                    filename=trento_ic_path)

    # Debug pwd
    print('Running freestream in...')
    print(os.getcwd())

    # Format trento data into initial conditions for freestream
    logging.info('Packaging trento initial conditions into array...')
    ic = toFsIc(initial_file=trento_ic_path, quiet=False)

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
    # create frzout HRG object (to be reused for all events) representing a hadron resonance gas at given temperature
    hrg_kwargs = dict(species='urqmd', res_width=True)
    hrg = frzout.HRG(t_end, **hrg_kwargs)

    # append switching energy density to hydro arguments
    # We use frzout's hrg class to compute an energy density based on the desired freezeout temperature
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

    # Determine maximum number of timesteps needed
    # This is the time it takes for a jet to travel across the plasma on its longest path at the speed of light
    maxTime = 2*rmax  # in fm --- equal to length to traverse in fm for c = 1 - 2x largest width of plasma
    logging.info('maxTime = %.3f fm', maxTime)

    # Dump the coarse run event data
    logging.info('Dumping coarse run hydro data')
    utilities.run_cmd(*['rm', 'viscous_14_moments_evo.dat'],
                      quiet=True)


    # Fine run
    logging.info('Running fine hydro...')
    hydro_dict = run_hydro(fs, event_size=rmax, grid_step=grid_step, tau_fs=tau_fs,
              hydro_args=hydro_args, time_step=time_step, maxTime=maxTime)

    logging.info('Event generation complete')
    logging.info('Analyzing event geometry...')

    # Add rmax to event_dataframe
    event_dataframe['rmax'] = rmax

    # Compute trento ic eccentricities
    event_dataframe['e2'] = np.sqrt(event_dataframe['e2_re'] ** 2 + event_dataframe['e2_im'] ** 2)
    event_dataframe['e3'] = np.sqrt(event_dataframe['e3_re'] ** 2 + event_dataframe['e3_im'] ** 2)
    event_dataframe['e4'] = np.sqrt(event_dataframe['e4_re'] ** 2 + event_dataframe['e4_im'] ** 2)
    event_dataframe['e5'] = np.sqrt(event_dataframe['e5_re'] ** 2 + event_dataframe['e5_im'] ** 2)

    # Compute flow coefficients v_n:
    # Create event surface object from hydro surface file dictionary
    event_surface = frzout.Surface(**hydro_dict, ymax=2)
    logging.info('%d freeze-out cells', len(event_surface))

    # Perform 1000 samples of frzout surface and take mean of computed v_2 and v_3
    logging.info('Sampling freezeout surface to compute v_n')
    v_2_array = np.array([])
    v_3_array = np.array([])
    num_frzout_samples = 1000
    for sample in np.arange(0, num_frzout_samples):
        
        # Sample particle production with frzout
        particles = frzout.sample(event_surface, hrg)
        
        # compute particle angle array
        particle_phi_array = np.array([])
        for part in particles:
            p = part['p']  # a single particle's position vector
            px = float(p[1])
            py = float(p[2])
            phi = np.arctan2(py, px)  # angle in xy-plane
            particle_phi_array = np.append(particle_phi_array, phi)
        
        # Compute flow vectors
        q2 = flow.qn(particle_phi_array, 2)
        q3 = flow.qn(particle_phi_array, 3)
        q4 = flow.qn(particle_phi_array, 4)
        
        # Compute cumulant
        vnk = flow.Cumulant(len(particle_phi_array), q2=q2, q3=q3, q4=q4)
        
        # Compute flow coefficients v_2{2} and v_3{2}
        v_2 = vnk.flow(2, 2, imaginary='negative')
        v_3 = vnk.flow(3, 2, imaginary='negative')
    
        v_2_array = np.append(v_2_array, v_2)
        v_3_array = np.append(v_3_array, v_3)
    
    v_2 = np.mean(v_2_array)
    v_3 = np.mean(v_3_array)

    logging.info('Flow coefficients computed using {} samples of frzout surface'.format(num_frzout_samples))
    logging.info('v_2 = {}'.format(v_2))
    logging.info('v_3 = {}'.format(v_3))

    event_dataframe['v_2'] = v_2
    event_dataframe['v_3'] = v_3

    logging.info('Event geometry analysis complete')


    if get_rmax is True:
        return event_dataframe, rmax
    else:
        return event_dataframe


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
    points = np.transpose(np.array([t_coords, x_coords, y_coords]), (2, 1, 0))

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
    points = np.transpose(np.array([t_coords, x_coords, y_coords]), (2, 1, 0))

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
        pointArray = cube_random(num = batch, boxSize=gridWidth, maxProb=maxTemp ** 6)

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
    return np.array([0,0,0])


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


# Function to generate new optical glauber event callable functions
def optical_glauber(R=7.5, b=7.5, phi=0, T0=1, U0=1):
    # Calculate ellipse height and width from ion radius (R) and impact parameter (b).
    W = 2 * R - b
    H = np.sqrt(4 * R ** 2 - b ** 2)

    # Calculate event multiplicity and eccentricity
    mult = 2*H*W*np.pi  # Integral of the 2D Gaussian for the temperature profile.
    e2 = np.sqrt(1 - (W ** 2 / H ** 2))  # sqrt(1 - (semi-minor^2 / semi-major^2))

    # Get cosine and sine of phi as fixed constants.
    cos_fac = np.cos(phi)
    sin_fac = np.sin(phi)

    analytic_t = lambda t, x, y: T0 * np.exp(
        - ((x * cos_fac + y * sin_fac) ** 2 / (2 * W ** 2)) - ((-x * sin_fac + y * cos_fac) ** 2 / (2 * H ** 2)))

    analytic_ux = lambda t, x, y: U0 * np.sqrt(W * H) * np.exp(
        - ((cos_fac * x + sin_fac * y) ** 2 / (2 * W ** 2)) - ((-sin_fac * x + cos_fac * y) ** 2 / (2 * H ** 2))) * (
                                              ((cos_fac * x + sin_fac * y) * cos_fac / W ** 2) - (
                                                  (-sin_fac * x + cos_fac * y) * sin_fac / H ** 2))

    analytic_uy = lambda t, x, y: U0 * np.sqrt(W * H) * np.exp(
        - ((cos_fac * x + sin_fac * y) ** 2 / (2 * W ** 2)) - ((-sin_fac * x + cos_fac * y) ** 2 / (2 * H ** 2))) * (
                                              ((cos_fac * x + sin_fac * y) * sin_fac / W ** 2) + (
                                                  (-sin_fac * x + cos_fac * y) * cos_fac / H ** 2))

    # To generate an event object from these optical glauber functions:
    #og_event = functional_plasma(temp_func=analytic_t, x_vel_func=analytic_ux, y_vel_func=analytic_uy)

    return analytic_t, analytic_ux, analytic_uy, mult, e2


# Function to generate new optical glauber event callable functions with log(mult) suppressed temps.
def optical_glauber_logT(R=7.5, b=7.5, phi=0, T0=1, U0=1):
    # Calculate ellipse height and width from ion radius (R) and impact parameter (b).
    W = 2 * R - b
    H = np.sqrt(4 * R ** 2 - b ** 2)

    # Calculate event multiplicity and eccentricity
    mult = 2*H*W*np.pi  # Integral of the 2D Gaussian for the temperature profile.
    e2 = np.sqrt(1 - (W ** 2 / H ** 2))  # sqrt(1 - (semi-minor^2 / semi-major^2))

    # Set temperature normalization
    T0 = T0 * np.log(mult)

    # Get cosine and sine of phi as fixed constants.
    cos_fac = np.cos(phi)
    sin_fac = np.sin(phi)

    analytic_t = lambda t, x, y: T0 * np.exp(
        - ((x * cos_fac + y * sin_fac) ** 2 / (2 * W ** 2)) - ((-x * sin_fac + y * cos_fac) ** 2 / (2 * H ** 2)))

    analytic_ux = lambda t, x, y: U0 * np.sqrt(W * H) * np.exp(
        - ((cos_fac * x + sin_fac * y) ** 2 / (2 * W ** 2)) - ((-sin_fac * x + cos_fac * y) ** 2 / (2 * H ** 2))) * (
                                              ((cos_fac * x + sin_fac * y) * cos_fac / W ** 2) - (
                                                  (-sin_fac * x + cos_fac * y) * sin_fac / H ** 2))

    analytic_uy = lambda t, x, y: U0 * np.sqrt(W * H) * np.exp(
        - ((cos_fac * x + sin_fac * y) ** 2 / (2 * W ** 2)) - ((-sin_fac * x + cos_fac * y) ** 2 / (2 * H ** 2))) * (
                                              ((cos_fac * x + sin_fac * y) * sin_fac / W ** 2) + (
                                                  (-sin_fac * x + cos_fac * y) * cos_fac / H ** 2))

    # To generate an event object from these optical glauber functions:
    #og_event = functional_plasma(temp_func=analytic_t, x_vel_func=analytic_ux, y_vel_func=analytic_uy)

    return analytic_t, analytic_ux, analytic_uy, mult, e2


# Function to generate new optical glauber event callable functions
def optical_glauber_new(R=7.5, b=7.5, phi=0, T0=1, U0=1):
    # Calculate ellipse height and width from ion radius (R) and impact parameter (b).
    W = 2 * R - b
    H = np.sqrt(4 * R ** 2 - b ** 2)

    # Calculate event multiplicity and eccentricity
    mult = 2*H*W*np.pi  # Integral of the 2D Gaussian for the temperature profile.
    e2 = np.sqrt(1 - (W ** 2 / H ** 2))  # sqrt(1 - (semi-minor^2 / semi-major^2))

    # Set temperature normalizations
    T0 = T0 * np.log(mult)

    # Get cosine and sine of phi as fixed constants.
    cos_fac = np.cos(phi)
    sin_fac = np.sin(phi)

    analytic_t = lambda t, x, y: T0 * np.exp(
        - ((x * cos_fac + y * sin_fac) ** 2 / (2 * W ** 2)) - ((-x * sin_fac + y * cos_fac) ** 2 / (2 * H ** 2)))

    analytic_ux = lambda t, x, y: U0 * np.sqrt(W * H) * np.exp(- ((cos_fac * x + sin_fac * y) ** 2 / (6 * W ** 2)) - (
                ((-sin_fac * (x) + cos_fac * y)) ** 2 / (6 * H ** 2))) * (
                                              (((cos_fac * x + sin_fac * y)) * cos_fac / W ** 2) - (
                                                  (-sin_fac * x + cos_fac * y) * sin_fac / H ** 2))

    analytic_uy = lambda t, x, y: U0 * np.sqrt(W * H) * np.exp(
        - ((cos_fac * x + sin_fac * y) ** 2 / (6 * W ** 2)) - ((-sin_fac * x + cos_fac * y) ** 2 / (6 * H ** 2))) * (
                                              ((cos_fac * x + sin_fac * y) * sin_fac / W ** 2) + (
                                                  (-sin_fac * x + cos_fac * y) * cos_fac / H ** 2))

    # To generate an event object from these optical glauber functions:
    #og_event = functional_plasma(temp_func=analytic_t, x_vel_func=analytic_ux, y_vel_func=analytic_uy)

    return analytic_t, analytic_ux, analytic_uy, mult, e2


# Function to select initial jet p_T based on 1/(p_T^4) distribution
def jet_pT_1opT4():
    rng = np.random.default_rng()
    while True:
        chosen_e = rng.uniform(config.jet.MIN_JET_ENERGY, 20)
        chosen_prob = rng.uniform(0, 1)
        if chosen_e > config.jet.MIN_JET_ENERGY and chosen_prob < (1 / chosen_e ** 4):
            break

    return chosen_e


# Function to select initial jet p_T based on jet spectra in sqrt(s)_{NN} = 200 GeV
# collisions at RHIC, as told by pi^0 production in pp + estimated cold nuclear matter effects
# https://inspirehep.net/literature/836952 - Cross-sections
# https://inspirehep.net/literature/595058 - Cold nuclear matter effects envelope
def jet_pT_RHIC():
    # Full data
    #pT_Domain = [1.25, 1.75, 2.25, 2.75, 3.5, 4.5, 5.5, 6.5, 7.5, 9, 11, 13, 15.5]
    #cross_sections = [0.259, 0.0483, 0.0119, 0.00364, 0.000438, 6.72E-05, 1.40E-05, 3.73E-06,
    #                  1.18E-06, 3.08E-07, 7.92E-08, 2.13E-08, 5.11E-09]
    # Cross sections for pi^0 production in pp
    pT_domain = [2.25, 2.75, 3.5, 4.5, 5.5, 6.5, 7.5, 9, 11, 13, 15.5]
    cross_sections = [0.0119, 0.00364, 0.000438, 6.72E-05, 1.40E-05, 3.73E-06,
                      1.18E-06, 3.08E-07, 7.92E-08, 2.13E-08, 5.11E-09]

    # Cold nuclear matter envelope on cross-sections
    # THESE ARE EYEBALLED NUMBERS. DO NOT TRUST THESE!!!
    env_pT_domain = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
    env = [1.15, 1.6, 1.3, 1.2, 1.1, 1.05, 0.98, 0.9, 0.88, 0.85, 0.8, 0.78, 0.76]

    # Interpolate envelope
    env_interpolator = interpolate.interp1d(env_pT_domain, env)

    # Apply envelope to cross-sections
    env_on_cross_sections = cross_sections * env_interpolator(pT_domain)

    # Interpolate enveloped cross-sections
    interpolator = interpolate.interp1d(pT_domain, env_on_cross_sections)

    rng = np.random.default_rng()
    while True:
        chosen_pT = rng.uniform(config.jet.MIN_JET_ENERGY, pT_domain[-1])
        chosen_prob = rng.uniform(0, np.amax(cross_sections))
        try:
            prob_pT = interpolator(chosen_pT)  # Function for pT probability
        except ValueError:
            # Do something else, throw a fit!
            logging.info('Problem sampling jet p_T spectrum!!!')
            prob_pT = chosen_pT * 1  # Function for pT probability

        if chosen_pT >= config.jet.MIN_JET_ENERGY and chosen_prob < prob_pT:
            break

    return chosen_pT

# Function to sample realistic cross-sections including cold nuclear matter effects
# to select jet pilot parton type and p_T.
def jet_sample_LHC(cent=None):
    #max_jet_pT_index = int(config.jet.MAX_JET_ENERGY * 100)  # Could use to limit data loaded

    # Select centrality bin fileset
    # Load proper file

    # Load file directly if in project root directory
    project_path = os.path.dirname(os.path.realpath(__file__))
    if cent is None:
        file_path = project_path + '/jet_cross_sections/rhic_calc.aa_cen_cron1.5_eloss0.5100GeV.out'
    else:
        file_path = project_path + '/jet_cross_sections/rhic_calc.aa_cen_cron1.5_eloss0.5100GeV.out'
    while True:
        try:
            cs_data = pd.read_table(file_path, header=None, delim_whitespace=True, dtype=np.float64,
                                    names=['pT', 'g', 'd', 'dbar', 'u', 'ubar', 's', 'sbar'], skiprows=1)
            logging.info('Found and read cross-section file...')
        except FileNotFoundError:
            logging.info('Failed to find cross-section file...')
        break

    # Cast to numpy arrays for interpolation
    pT_domain = cs_data['pT'].to_numpy()
    sigma_u = cs_data['u'].to_numpy()
    sigma_ubar = cs_data['ubar'].to_numpy()
    sigma_d = cs_data['d'].to_numpy()
    sigma_dbar = cs_data['dbar'].to_numpy()
    sigma_s = cs_data['s'].to_numpy()
    sigma_sbar = cs_data['sbar'].to_numpy()
    sigma_g = cs_data['g'].to_numpy()

    # Compute maximum of cross-sections
    cross_section_max = np.amax(
        [np.amax(sigma_u), np.amax(sigma_ubar), np.amax(sigma_d),
         np.amax(sigma_dbar), np.amax(sigma_s),
         np.amax(sigma_sbar), np.amax(sigma_g)])

    # Interpolate cross-section spectra
    logging.info('Interpolating cross section data')
    sigma_u_int = interpolate.interp1d(pT_domain, sigma_u)
    sigma_ubar_int = interpolate.interp1d(pT_domain, sigma_ubar)
    sigma_d_int = interpolate.interp1d(pT_domain, sigma_d)
    sigma_dbar_int = interpolate.interp1d(pT_domain, sigma_dbar)
    sigma_s_int = interpolate.interp1d(pT_domain, sigma_s)
    sigma_sbar_int = interpolate.interp1d(pT_domain, sigma_sbar)
    sigma_g_int = interpolate.interp1d(pT_domain, sigma_g)

    logging.info('Sampling cross-sections...')
    rng = np.random.default_rng()
    while True:
        # Randomly select a point in the phase space and a corresponding probability measure
        chosen_pT = rng.uniform(config.jet.MIN_JET_ENERGY, config.jet.MAX_JET_ENERGY)
        chosen_prob = rng.uniform(0, cross_section_max)
        chosen_pilot = rng.choice(['s', 'sbar', 'u', 'ubar', 'd', 'dbar', 'g'])

        # Check corresponding real probability of this point
        try:
            if chosen_pilot == 'u':
                prob_pT = sigma_u_int(chosen_pT)
            elif chosen_pilot == 'ubar':
                prob_pT = sigma_ubar_int(chosen_pT)
            elif chosen_pilot == 'd':
                prob_pT = sigma_d_int(chosen_pT)
            elif chosen_pilot == 'dbar':
                prob_pT = sigma_dbar_int(chosen_pT)
            elif chosen_pilot == 's':
                prob_pT = sigma_s_int(chosen_pT)
            elif chosen_pilot == 'sbar':
                prob_pT = sigma_sbar_int(chosen_pT)
            elif chosen_pilot == 'g':
                prob_pT = sigma_g_int(chosen_pT)
            else:
                prob_pT = 0

        except ValueError:
            # Do something else, throw a fit!
            logging.info('Problem sampling jet p_T spectrum!!!')
            prob_pT = chosen_pT * 1  # Function for pT probability

        # If chosen probability is less than true probability, accept this point
        if chosen_prob < prob_pT:
            break

    return chosen_pilot, chosen_pT

# Function to use importance sampling to select a uniform-random jet pilot and pT
# with an associated weight drawn from the ratio of the pilot's cross-section distribution
# and the maximum cross-section value.
def jet_IS_LHC(cent=None, num_samples=1):
    # Select centrality bin fileset
    # Load proper file

    # Load file directly if in project root directory
    project_path = os.path.dirname(os.path.realpath(__file__))
    if cent is None:
        file_path = project_path + '/jet_cross_sections/rhic_calc.aa_cen_cron1.5_eloss0.5100GeV.out'
    else:
        file_path = project_path + '/jet_cross_sections/rhic_calc.aa_cen_cron1.5_eloss0.5100GeV.out'
    while True:
        try:
            cs_data = pd.read_table(file_path, header=None, delim_whitespace=True, dtype=np.float64,
                                    names=['pT', 'g', 'd', 'dbar', 'u', 'ubar', 's', 'sbar'], skiprows=1)
            logging.info('Found and read cross-section file...')
        except FileNotFoundError:
            logging.info('Failed to find cross-section file...')
        break

    # Cast to numpy arrays for interpolation
    pT_domain = cs_data['pT'].to_numpy()
    sigma_u = cs_data['u'].to_numpy()
    sigma_ubar = cs_data['ubar'].to_numpy()
    sigma_d = cs_data['d'].to_numpy()
    sigma_dbar = cs_data['dbar'].to_numpy()
    sigma_s = cs_data['s'].to_numpy()
    sigma_sbar = cs_data['sbar'].to_numpy()
    sigma_g = cs_data['g'].to_numpy()

    # Compute maximum of cross-sections
    cross_section_max = np.amax(
        [np.amax(sigma_u), np.amax(sigma_ubar), np.amax(sigma_d),
         np.amax(sigma_dbar), np.amax(sigma_s),
         np.amax(sigma_sbar), np.amax(sigma_g)])

    # Interpolate cross-section spectra
    logging.info('Interpolating cross section data')
    sigma_u_int = interpolate.interp1d(pT_domain, sigma_u)
    sigma_ubar_int = interpolate.interp1d(pT_domain, sigma_ubar)
    sigma_d_int = interpolate.interp1d(pT_domain, sigma_d)
    sigma_dbar_int = interpolate.interp1d(pT_domain, sigma_dbar)
    sigma_s_int = interpolate.interp1d(pT_domain, sigma_s)
    sigma_sbar_int = interpolate.interp1d(pT_domain, sigma_sbar)
    sigma_g_int = interpolate.interp1d(pT_domain, sigma_g)

    logging.info('Sampling cross-sections...')
    rng = np.random.default_rng()
    chosen_pT_array = np.array([])
    chosen_pilot_array = np.array([])
    chosen_weight_array = np.array([])
    while len(chosen_weight_array) < num_samples:
        # Randomly select a point in the phase space and a corresponding probability measure
        chosen_pT = rng.uniform(config.jet.MIN_JET_ENERGY, config.jet.MAX_JET_ENERGY)
        chosen_pilot = rng.choice(['s', 'sbar', 'u', 'ubar', 'd', 'dbar', 'g'])

        # Check corresponding real probability of this point
        try:
            if chosen_pilot == 'u':
                chosen_weight = sigma_u_int(chosen_pT)/cross_section_max
            elif chosen_pilot == 'ubar':
                chosen_weight = sigma_ubar_int(chosen_pT)/cross_section_max
            elif chosen_pilot == 'd':
                chosen_weight = sigma_d_int(chosen_pT)/cross_section_max
            elif chosen_pilot == 'dbar':
                chosen_weight = sigma_dbar_int(chosen_pT)/cross_section_max
            elif chosen_pilot == 's':
                chosen_weight = sigma_s_int(chosen_pT)/cross_section_max
            elif chosen_pilot == 'sbar':
                chosen_weight = sigma_sbar_int(chosen_pT)/cross_section_max
            elif chosen_pilot == 'g':
                chosen_weight = sigma_g_int(chosen_pT)/cross_section_max
            else:
                chosen_weight = 0

        except ValueError:
            # Do something else, throw a fit!
            logging.warning('Problem sampling jet p_T spectrum!!!')
            chosen_weight = 1  # Function for pT probability

        chosen_pT_array = np.append(chosen_pT_array, chosen_pT)
        chosen_pilot_array = np.append(chosen_pilot_array, chosen_pilot)
        chosen_weight_array = np.append(chosen_weight_array, chosen_weight)

    if num_samples == 1:
        return chosen_pilot_array[0], chosen_pT_array[0], chosen_weight_array[0]
    else:
        return chosen_pilot_array, chosen_pT_array, chosen_weight_array
