import argparse
from contextlib import contextmanager
import datetime
from itertools import chain, groupby, repeat
import logging
import math
import os
import pickle
import shutil
import signal
import subprocess
import sys
import tempfile

import numpy as np
import h5py

# Command to run process in the terminal
# Stolen and modified from DukeQCD "run-events.py":
# https://github.com/Duke-QCD/hic-eventgen
def run_cmd(*args):
    quiet = False
    """
    Run and log a subprocess.
    """
    cmd = ' '.join(args)
    logging.info('running command: %s', cmd)
    processName = str(args[0])

    try:
        proc = subprocess.run(
            cmd.split(), check=True,
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
            proc.stdout.strip('\n')
        )

        if not quiet:
            # Attempt to print the output from the subprocess.
            print('------------- {} Output ----------------'.format(processName))
            print('format: event_number impact_param npart mult e2 e3 e4 e5')
            print('exit status:\n', proc.returncode)
            print('stdout:\n', proc.stdout)
            print('stderr:\n', proc.stderr)
            print('---------- {} Output End ---------------'.format(processName))

        return proc


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