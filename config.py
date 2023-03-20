import yaml
import os

############################################################
# See config.yml for descriptions of configuration options #
############################################################
# Get location of config.py and config.yml
project_path = os.path.dirname(os.path.realpath(__file__))

# Read config file and parse settings
with open(project_path + '/config.yml', 'r') as ymlfile:
    # Note the usage of yaml.safe_load()
    # Using yaml.load() exposes the system to running any Python commands in the config file.
    # That is unnecessary risk!!!
    cfg = yaml.safe_load(ymlfile)


# Event by event sampling configuration
class EBE:
    NUM_EVENTS = int(cfg['mode']['NUM_EVENTS'])
    NUM_SAMPLES = int(cfg['mode']['NUM_SAMPLES'])


# Mode configuration
class mode:
    RANDOM_EVENT = bool(cfg['mode']['RANDOM_EVENT'])
    RANDOM_ANGLE = bool(cfg['mode']['RANDOM_ANGLE'])
    VARY_POINT = bool(cfg['mode']['VARY_POINT'])
    WEIGHT_POINT = bool(cfg['mode']['WEIGHT_POINT'])
    KEEP_EVENT = bool(cfg['mode']['KEEP_EVENT'])


class transport:
    GRID_STEP = float(cfg['transport']['GRID_STEP'])
    TIME_STEP = float(cfg['transport']['TIME_STEP'])
    GRID_MAX_TARGET = float(cfg['transport']['GRID_MAX_TARGET'])

    class trento:
        NORM = float(cfg['trento']['NORM'])
        PROJ1 = str(cfg['trento']['PROJ1'])
        PROJ2 = str(cfg['trento']['PROJ2'])
        NUCLEON_WIDTH = float(cfg['trento']['NUCLEON_WIDTH'])
        CROSS_SECTION = float(cfg['trento']['CROSS_SECTION'])

        try:
            BMIN = float(cfg['trento']['BMIN'])
        except ValueError:
            BMIN = None
        try:
            BMAX = float(cfg['trento']['BMAX'])
        except ValueError:
            BMAX = None

    class hydro:
        TAU_FS = float(cfg['transport']['TAU_FS'])
        T_END = float(cfg['transport']['T_END'])
        T_HRG = float(cfg['transport']['T_HRG'])
        T_UNHYDRO = float(cfg['transport']['T_UNHYDRO'])

    class afterburner:
        USE = False


# Jet configuration
class jet:
    JET_ENERGY = int(cfg['jet']['JET_ENERGY'])
    E_FLUCT = bool(cfg['jet']['E_FLUCT'])
    E_IS = bool(cfg['jet']['E_IS'])
    MIN_JET_ENERGY = float(cfg['jet']['MIN_JET_ENERGY'])
    MAX_JET_ENERGY = float(cfg['jet']['MAX_JET_ENERGY'])
    TAU = float(cfg['jet']['TAU'])
    RECORD = bool(cfg['jet']['RECORD'])



# Global constants
class constants:
    G = float(cfg['global_constants']['G'])
    G_MU = float(cfg['global_constants']['G_MU'])
    K_BBMG = float(cfg['global_constants']['K_BBMG'])
    K_DRIFT = float(cfg['global_constants']['K_DRIFT'])
