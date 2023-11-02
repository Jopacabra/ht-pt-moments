import pythia8
import numpy as np
import pandas as pd

# Function to generate a pp hard scattering at sqrt(s) = 5.02 TeV
def scattering():
    ############
    # Settings #
    ############
    y_res = 0.5

    # Generate scattering id
    scattering_id = int(np.random.uniform(0, 1000000000000))

    ############################
    # Set up custom user hooks #
    ############################
    # Write own derived UserHooks class.
    class MyUserHooks(pythia8.UserHooks):

        # Constructor to make a user hook
        def __init__(self):
            pythia8.UserHooks.__init__(self)

        # Allow process cross section to be modified...
        def canModifySigma(self):
            return True

        # ...which gives access to the event at the trial level, before selection.
        def multiplySigmaBy(self, sigmaProcessPtr, phaseSpacePtr, inEvent):

            # All events should be 2 -> 2, kill them if not.
            if sigmaProcessPtr.nFinal() != 2: return 0.

            # Here we do not modify 2 -> 2 cross sections.
            return 1.

        # Allow a veto after process selection.
        def canVetoProcessLevel(self):
            return True

        # Veto events that do not fit the desired requirements
        def doVetoProcessLevel(self, process):
            # Get info
            info = pythia_process.infoPython()

            # Get only events at mid-rapidity, within my chosen y_res
            if np.abs(info.y()) < y_res:  # and np.abs(chosen_pt -np.abs(info.pTHat())) < pt_hat_res:
                return False  # Do not veto the event
            else:
                return True  # Veto the event


    #################
    # Set up Pythia #
    #################
    pythia_process = pythia8.Pythia("", False)  # Print header = False

    # Use seed based on time
    pythia_process.readString("Random:setSeed = on")
    pythia_process.readString("Random:seed = 0")

    # # Set beam energy - in GeV
    pythia_process.readString("Beams:eCM = 5020.")

    # Set particles in each beam - defaults to proton (2212), if nothing set
    pythia_process.readString("Beams:idA = 2212")
    pythia_process.readString("Beams:idB = 2212")

    # Only do the parton level stuff
    pythia_process.readString("ProcessLevel:all = on")
    pythia_process.readString("PartonLevel:all = off")
    pythia_process.readString("HadronLevel:all = off")

    # Turn on all hard QCD processes
    pythia_process.readString("HardQCD:all = on")

    # Turn off all showers
    # pythia_process.readString("PartonLevel:MPI = off")
    # pythia_process.readString("PartonLevel:ISR = off")
    # pythia_process.readString("PartonLevel:FSR = off")
    # pythia_process.readString("PartonLevel:FSRinProcess = off")
    # pythia_process.readString("PartonLevel:FSRinResonances = off")
    # pythia_process.readString("PartonLevel:earlyResDec = off")
    # pythia_process.readString("PartonLevel:Remnants = on")

    # Set a phase space cut for particle pT.
    '''
    The HardQCD 2->2 processes are divergent as pT -> 0, so we need some cut here.
    Note that this parton-level cut does not necessarily put a cut on jet phase space.
    intermediate parton showers, MPIs, hadronization effects, and jet finders will distort the original simple process
    '''
    pythia_process.readString("PhaseSpace:pTHatMin = 1")  # Phase space cuts are on hard process pTHat
    pythia_process.readString("PhaseSpace:pTHatMax = 100")

    # Here we bias the selection of pTHat for the process by a given power of pTHat (Here pTHat^4)
    # This is more or less equivalent to sampling from a uniform distribution in pTHat
    # and recording an appropriate true pTHat-dependent weight from a known weight distribution
    pythia_process.readString("PhaseSpace:bias2Selection = on")
    pythia_process.readString("PhaseSpace:bias2SelectionPow = 4")

    # Set up to do a user veto and send it in.
    myUserHooks = MyUserHooks()
    pythia_process.setUserHooksPtr(myUserHooks)

    # Tell Pythia to "do the thing" (run with the configurations above)
    pythia_process.init()

    ####################################
    # Run the event generation routine #
    ####################################

    # Generate event. Skip if error.
    """
    This is the event loop. When we call pythia.next(), we generate the next event. If this returns "False",
    an error occured, so we try again.

    We then check for a satisfactory particle. If it exists, we accept the event. If not, we generate a new one.
    """

    failed_events = 0
    success = False
    while not success:
        if not pythia_process.next():  # Note: Calling pythia.next() generates the next event in with the pythia object.
            failed_events += 1
        else:
            break  # Do not check output particles
            # for particle in pythia_process.process:
            #     if particle.status() > 0:
            #         if np.abs(float(particle.y())) < (y_res / 2):
            #             # if np.abs(particle.id()) == 3 or np.abs(particle.id()) == 2 or np.abs(particle.id()) == 1  \
            #             #         or particle.id() == 21:
            #             # if np.abs(chosen_pt - np.abs(particle.pT())) < (pt_res/2):
            #             id = particle.id()
            #             if id == chosen_id:
            #                 success = True
            # failed_events += 1

    ################################
    # Package and output particles #
    ################################
    weight = pythia_process.infoPython().weight()
    particles = pd.DataFrame({})

    for particle in pythia_process.process:
        if particle.id() != 90:
            properties = pd.DataFrame(
                {
                    'id': [int(particle.id())],
                    'status': [int(particle.status())],
                    'mother1': [int(particle.mother1())],
                    'mother2': [int(particle.mother2())],
                    'daughter1': [int(particle.daughter1())],
                    'daughter2': [int(particle.daughter2())],
                    'col': [int(particle.col())],
                    'acol': [int(particle.acol())],
                    'px': [particle.px()],
                    'py': [particle.py()],
                    'pz': [particle.pz()],
                    'pt': [particle.pT()],
                    'e': [particle.e()],
                    'm': [particle.m()],
                    'scaleIn': [particle.scale()]
                }
            )

            particles = pd.concat([particles, properties], axis=0)

    return particles, weight


# Function to hadronize a pair of particles
def fragment(jet1, jet2, process_dataframe, weight):
    # Settings
    y_res = 1
    max_had_runs = 10000

    #########################
    # Assign colors to jets #
    #########################

    # Get particle ids
    id1 = jet1.id
    id2 = jet2.id

    remnant = False
    remnant2 = False
    # Choose colors so as to get a color singlet
    # Add a third particle as a beam remnant to get a color singlet, if necessary
    if id1 == 21 and id2 == 21:
        # A pair of gluons
        # Particles just get opposite colors and anticolors
        col1 = 101
        acol1 = 102
        col2 = 102
        acol2 = 101
    elif (3.1 > id1 > 0) and (-3.1 < id2 < 0):
        # Quark antiquark pair
        col1 = 101
        acol1 = 0
        col2 = 0
        acol2 = 101
    elif (-3.1 < id1 < 0) and (3.1 > id2 > 0):
        # Antiquark quark pair
        col1 = 0
        acol1 = 101
        col2 = 101
        acol2 = 0
    elif (-3.1 < id1 < 0) and id2 == 21:
        # antiquark gluon pair
        remnant = True
        rem_col = 102
        rem_acol = 0
        col1 = 0
        acol1 = 101
        col2 = 101
        acol2 = 102
    elif id1 == 21 and (-3.1 < id2 < 0):
        # gluon antiquark pair
        remnant = True
        rem_col = 102
        rem_acol = 0
        col1 = 101
        acol1 = 102
        col2 = 0
        acol2 = 101
    elif (3.1 > id1 > 0) and id2 == 21:
        # quark gluon pair
        remnant = True
        rem_col = 0
        rem_acol = 102
        col1 = 101
        acol1 = 0
        col2 = 102
        acol2 = 101
    elif id1 == 21 and (3.1 > id2 > 0):
        # gluon quark pair
        remnant = True
        rem_col = 0
        rem_acol = 101
        col1 = 101
        acol1 = 102
        col2 = 102
        acol2 = 0
    elif (3.1 > id1 > 0) and (3.1 > id2 > 0):
        # quark quark pair
        remnant = True
        rem_col = 0
        rem_acol = 101
        rem2_col = 0
        rem2_acol = 102
        col1 = 101
        acol1 = 0
        col2 = 102
        acol2 = 0
    elif (-3.1 < id1 < 0) and (-3.1 < id2 < 0):
        # antiquark antiquark pair
        remnant = True
        rem_col = 101
        rem_acol = 0
        rem2_col = 102
        rem2_acol = 0
        col1 = 0
        acol1 = 101
        col2 = 0
        acol2 = 102

    col_array = np.array([col1, col2])
    acol_array = np.array([acol1, acol2])
    if remnant:
        col_array = np.append(col_array, rem_col)
        acol_array = np.append(acol_array, rem_acol)
    if remnant2:
        col_array = np.append(col_array, rem2_col)
        acol_array = np.append(acol_array, rem2_acol)

    ############################################
    # Set up Pythia instance for hadronization #
    ############################################

    # Instantiate Pythia
    pythia_had = pythia8.Pythia("", False)  # Print header = False

    # Use seed based on time
    pythia_had.readString("Random:setSeed = on")
    pythia_had.readString("Random:seed = 0")

    # Only do the hadron level stuff
    pythia_had.readString("ProcessLevel:all = off")
    pythia_had.readString("PartonLevel:all = off")
    pythia_had.readString("HadronLevel:all = on")

    # Don't allow pi^0 to decay:
    pythia_had.readString("111:mayDecay = off")

    # Allow color reconnection in hadronization
    # pythia_had.readString("ColourReconnection:forceHadronLevelCR = on")

    # Turn off event checks that enforce conservation of momentum in the event
    pythia_had.readString("Check:event = off")

    # Tell Pythia to "do the thing" (run with the configurations above)
    pythia_had.init()

    #################################
    # Run the hadronization routine #
    #################################
    # We repeatedly hadronize until we come out with a satisfactory pion, saving info on the statistical weight
    accepted = False
    total_pions = 0
    total_had_runs = 0
    while total_had_runs < max_had_runs:

        # Clear the event
        pythia_had.event.reset()

        # Add in edited particles
        part_i = -1
        i = 0
        for index, particle in process_dataframe.iterrows():  #np.array([pythia_process.process[5], pythia_process.process[6]]):
            part_i += 1
            # Append to the hadronization event if the particle exists in the final state
            '''
            Lots of important things to note here...
            
            1. We must append a scale at which the particle was produced, otherwise parton showers and 
                hadronization won't occur
            2. We kill all of the z momentum, so we need to recompute the particle energy as on-shell
            '''
            # Importantly,
            # and hadronization cannot occur!!!

            # Add source particles in
            if particle['id'] != 90 and particle['status'] < 1:
                pythia_had.event.append(id=int(particle['id']), status=int(particle['status']),
                                        mother1=int(particle['mother1']), mother2=int(particle['mother2']),
                                        daughter1=int(particle['daughter1']), daughter2=int(particle['daughter2']),
                                        col=int(particle['col']), acol=int(particle['acol']),
                                        px=float(particle['px']), py=float(particle['py']), pz=float(particle['pz']),
                                        e=float(particle['e']), m=float(particle['m']),
                                        scaleIn=float(particle['scaleIn']))
            # Add jet seed particles back in, with momentum modifications
            elif particle['id'] != 90 and particle['status'] > 1:
                if i == 4:
                    pythia_had.event.append(id=int(particle['id']), status=int(particle['status']),
                                            mother1=int(particle['mother1']), mother2=int(particle['mother2']),
                                            daughter1=int(particle['daughter1']), daughter2=int(particle['daughter2']),
                                            col=int(col_array[0]), acol=int(acol_array[0]),
                                            px=float(jet1.p_x), py=float(jet1.p_y), pz=0,
                                            e=float(np.sqrt(jet1.p_x**2 + jet1.p_y**2 + particle['m']**2)), m=float(particle['m']),
                                            scaleIn=float(particle['scaleIn']))
                elif i == 5:
                    pythia_had.event.append(id=int(particle['id']), status=int(particle['status']),
                                            mother1=int(particle['mother1']), mother2=int(particle['mother2']),
                                            daughter1=int(particle['daughter1']), daughter2=int(particle['daughter2']),
                                            col=int(col_array[1]), acol=int(acol_array[1]),
                                            px=float(jet2.p_x), py=float(jet2.p_y), pz=0,
                                            e=float(np.sqrt(jet2.p_x ** 2 + jet2.p_y ** 2 + particle['m'] ** 2)),
                                            m=float(particle['m']),
                                            scaleIn=float(particle['scaleIn']))
                    scaleIn_last = float(particle['scaleIn'])
            i += 1
        if remnant:
            if rem_col != 0:
                rem_id = np.random.default_rng().choice([2, 2, 1])
                if rem_id == 2:
                    rem_m = 0.0022
                else:
                    rem_m = 0.0047
            else:
                rem_id = np.random.default_rng().choice([-2, -2, -1])
                if rem_id == -2:
                    rem_m = 0.0022
                else:
                    rem_m = 0.0047
            pythia_had.event.append(id=int(rem_id), status=int(23),
                                    col=int(col_array[2]), acol=int(acol_array[2]),
                                    px=0, py=0, pz=10000,
                                    e=float(np.sqrt(0 ** 2 + 0 ** 2 + 10000 **2 + rem_m ** 2)),
                                    m=float(rem_m),
                                    scaleIn=float(scaleIn_last))
            i += 1
        if remnant2:
            if rem2_col != 0:
                rem2_id = np.random.default_rng().choice([2, 2, 1])
                if rem2_id == 2:
                    rem2_m = 0.0022
                else:
                    rem2_m = 0.0047
            else:
                rem2_id = np.random.default_rng().choice([-2, -2, -1])
                if rem2_id == -2:
                    rem2_m = 0.0022
                else:
                    rem2_m = 0.0047
            pythia_had.event.append(id=int(rem2_id), status=int(23),
                                    col=int(col_array[3]), acol=int(acol_array[3]),
                                    px=0, py=0, pz=-10000,
                                    e=float(np.sqrt(0 ** 2 + 0 ** 2 + 10000 **2 + rem2_m ** 2)),
                                    m=float(rem2_m),
                                    scaleIn=float(scaleIn_last))
            i += 1
        # part_i = -1
        # for particle in pythia_had.process:
        #     part_i += 1
        # # Force parton shower
        # # Set all particles allowed to shower
        # shower_pTmax = pythia_process.infoPython().pTHat()
        # pythia_had.forceTimeShower(iBeg=part_i-1, iEnd=part_i, pTmax=shower_pTmax)  #, nBranchMax=10)

        # List particles for debug
        pythia_had.event.list()

        # hadronize
        pythia_had.next()

        # List particles again for debug
        pythia_had.event.list()

        # Look for an acceptable pion
        hadron_accepted_px = np.array([])
        hadron_accepted_py = np.array([])
        hadron_accepted_pz = np.array([])
        hadron_accepted_y = np.array([])
        hadron_accepted_e = np.array([])
        hadron_accepted_pt = np.array([])
        hadron_accepted_id = np.array([])
        hadron_f_pt = np.array([])
        pions_f = 0
        for particle in pythia_had.event:
            if particle.status() > 0:  # Particle exists in the final state
                id = particle.id()
                if id == 111 or np.abs(id) == 211:  # Particle is a pion
                    pions_f += 1
                    hadron_y = particle.y()
                    hadron_pt = particle.pT()
                    hadron_f_pt = np.append(hadron_f_pt, np.abs(hadron_pt))
                    pions_f += 1
                    if np.abs(hadron_y) < 1:  # Particle is at mid-rapidity
                        if np.abs(hadron_pt) > 1:  # Particle is hard -- substantially above medium scale
                            accepted = True
                            hadron_accepted_px = np.append(hadron_accepted_px, particle.px())
                            hadron_accepted_py = np.append(hadron_accepted_py, particle.py())
                            hadron_accepted_pz = np.append(hadron_accepted_pz, particle.pz())
                            hadron_accepted_y = np.append(hadron_accepted_y, hadron_y)
                            hadron_accepted_e = np.append(hadron_accepted_e, particle.e())
                            hadron_accepted_pt = np.append(hadron_accepted_pt, np.abs(particle.pT()))
                            hadron_accepted_id = np.append(hadron_accepted_id, particle.id())

        # Count pions and runs to determine weight of the final pion
        total_pions += pions_f
        total_had_runs += 1

        if accepted:
            break


    hadrons = pd.DataFrame(
        {
            'id': hadron_accepted_id.astype(int),
            'px': hadron_accepted_px.astype(float),
            'py': hadron_accepted_py.astype(float),
            'pz': hadron_accepted_pz.astype(float),
            'pt': hadron_accepted_pt.astype(float),
            'y': hadron_accepted_y.astype(float),
            'e': hadron_accepted_e.astype(float),
            'weight': np.full_like(hadron_accepted_id, float(weight)).astype(float),
            'num_hrz': np.full_like(hadron_accepted_id, int(total_had_runs)).astype(int)
        })

    return hadrons

