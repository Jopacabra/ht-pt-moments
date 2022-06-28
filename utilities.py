import logging
import math
import os
import subprocess
import tempfile

import numpy as np


# Command to run process in the terminal
# Stolen and modified from DukeQCD "run-events.py":
# https://github.com/Duke-QCD/hic-eventgen
import pandas as pd


def run_cmd(*args, quiet=False):
    """
    Run and log a Subprocess.
    """
    cmd = ' '.join(args)
    logging.info('running command: %s', cmd)
    processName = str(args[0])

    try:
        proc = subprocess.Popen(
            cmd.split(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True
        )
    except subprocess.CalledProcessError as e:
        logging.error(
            'command failed with status %d:\n%s',
            e.returncode, e.output.strip('\n')
        )
        raise
    else:
        logging.debug(
            'command completed successfully:\n%s',
            proc.stdout
        )
        outputArray = np.array([])
        if not quiet:
            outputCopyStdout = proc.stdout
            outputCopyStderr = proc.stderr
            # Attempt to print the output from the trentoSubprocess.
            logging.info('------------- {} Output ----------------'.format(processName))
            logging.debug('exit status:\n', proc.returncode)
            logging.info('stdout:\n')
            for line in outputCopyStdout:
                logging.info(line)
                outputArray = np.append(outputArray, line)
            logging.debug('stderr:\n')
            while True:
                try:
                    line = outputCopyStderr.readline()
                except AttributeError:
                    break
                logging.debug(line)
                if not line:
                    break
            logging.info('----------------------------------------')

        return proc, outputArray


# Function to round up to specified number of decimals
def round_decimals_up(number: float, decimals: int = 2):
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor


# Function to round down to specified number of decimals
def round_decimals_down(number: float, decimals: int = 1):
    """
    Returns a value rounded down to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.floor(number)

    factor = 10 ** decimals
    return math.floor(number * factor) / factor


# Function to create empty results dataframe.
def resultsFrame():
    resultsDataframe = pd.DataFrame(
            {
                "eventNo": [],
                "jetNo": [],
                "pT_plasma": [],
                "pT_plasma_error": [],
                "pT_hrg": [],
                "pT_hrg_error": [],
                "pT_unhydro": [],
                "pT_unhydro_error": [],
                "k_moment": [],
                "deflection_angle_plasma": [],
                "deflection_angle_plasma_error": [],
                "deflection_angle_hrg": [],
                "deflection_angle_hrg_error": [],
                "deflection_angle_unhydro": [],
                "deflection_angle_unhydro_error": [],
                "shower_correction": [],
                "X0": [],
                "Y0": [],
                "theta0": [],
                "t_unhydro": [],
                "t_hrg": [],
                "time_total_plasma": [],
                "time_total_hrg": [],
                "time_total_unhydro": [],
                "initial_time": [],
                "final_time": [],
                "b": [],
                "npart": [],
                "mult": [],
                "e2_re": [],
                "e2_im": [],
                "e3_re": [],
                "e3_im": [],
                "e4_re": [],
                "e4_im": [],
                "e5_re": [],
                "e5_im": [],
                "seed": [],
                "cmd": [],
            }
        )

    return resultsDataframe


# Creates a temporary directory and moves to it.
# Returns tempfile.TemporaryDirectory object.
def tempDir(location=None):
    # Get current directory if no location supplied
    if location is None:
        location = os.getcwd()
    # Create and move to temp directory
    temp_dir = tempfile.TemporaryDirectory(prefix='JMA_', dir=str(location))
    print('Created temp directory {}'.format(temp_dir.name))
    os.chdir(temp_dir.name)

    return temp_dir
