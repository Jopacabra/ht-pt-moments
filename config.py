import yaml

# Read config file and parse settings
with open('config.yml', 'r') as ymlfile:
    # Note the usage of yaml.safe_load()
    # Using yaml.load() exposes the system to running any Python commands in the config file.
    # That is unnecessary risk!!!
    cfg = yaml.safe_load(ymlfile)

# Set scanning modes
RANDOM_EVENT = bool(cfg['mode']['RANDOM_EVENT'])  # Determines if we select a random event or iterate through each
RANDOM_ANGLE = bool(cfg['mode']['RANDOM_ANGLE'])  # Determines if we select a random angle or do a complete sweep
VARY_POINT = bool(cfg['mode']['VARY_POINT'])  # Determines if we vary the prod point or set it to (0,0)
WEIGHT_POINT = bool(cfg['mode'][
                        'WEIGHT_POINT'])  # Determines if we weight point selection by T^6 in event - Needs VARYPOINT.

# Set jet parameters
N = int(cfg['jet_parameters']['NUM_JETS'])  # Number of jets to produce
JET_ENERGY = int(cfg['jet_parameters']['JET_ENERGY'])  # Jet energy

# Set moment parameters
K = int(cfg['moment_parameters']['MOMENT_K'])  # Which k-moment to calculate
TEMP_CUTOFF = float(cfg['moment_parameters']['TEMP_CUTOFF'])  # Temp at which to set plasma integrand to 0

# Set all the global constants
G = float(cfg['global_constants']['G'])  # Coupling constant for strong interaction

# Set config settings

if RANDOM_ANGLE == False and VARY_POINT == False and RANDOM_EVENT == False:
    print('Calculating angular sweep of moments for central jets in each background...')
elif RANDOM_ANGLE == False and VARY_POINT == False and RANDOM_EVENT == True:
    print('Calculating angular sweep of moments for central jets in random backgrounds...')
elif RANDOM_ANGLE == True and VARY_POINT == True:
    print('Calculating a given number of random jets at random angles in random backgrounds from library...')
elif RANDOM_ANGLE == True and VARY_POINT == False:
    print('Calculating a given number of central jets at random angles in random backgrounds from library...')
elif RANDOM_ANGLE == False and VARY_POINT == True:
    print('Calculating angular sweep of random jets in random backgrounds from library...')
else:
    print('Input invalid mode!!!!!')
    print('RANDOM_ANGLE = ' + str(RANDOM_ANGLE))
    print('VARY_POINT = ' + str(VARY_POINT))
    print('RANDOM_EVENT = ' + str(RANDOM_EVENT))
    print('WEIGHT_POINT = ' + str(WEIGHT_POINT))