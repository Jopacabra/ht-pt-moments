import pythia8
import pythia
import numpy as np
import pandas as pd

#############################
# Create a pp vacuum shower #
#############################

particles, weight = pythia.scattering(pThatmin=1, pThatmax=100)

################################################
# Apply energy loss and drift to all particles #
################################################

medium = False
if medium:
    pass

#############################
# Hadronize modified shower #
#############################

event = pythia.pp_shower_hadronize(particles=particles)
