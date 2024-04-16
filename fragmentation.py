import numpy as np
import lhapdf

def frag(parton):
    # Get jet properties
    jet_pt = parton.p_T()
    jet_pid = parton.id

    # Set limits on fragmentation values
    z_min = 0.01
    z_max = 1

    # Start an rng instance
    rng = np.random.default_rng()

    # Define probability distribution of z values
    def frag_p_z(ff, pid, pt_part):
        # Compute unnormalized p(z)
        if pt_part < 1.14:
            pt_part = 1.14
        Q2 = pt_part ** 2  # Q^2 in Gev^2 -- equivalent to parton p_T^2
        p_z = lambda z: ff.xfxQ2(pid, z, Q2)
        return p_z

    # Load fragmentation function
    loaded_ff = lhapdf.mkPDF("JAM20-SIDIS_FF_hadron_nlo")

    # Define probability distribution of z values for given p_T^{part} and max prob.
    p_z = frag_p_z(ff=loaded_ff, pid=jet_pid, pt_part=jet_pt)

    # Find maximum value of P(z)
    z_val_array = np.arange(z_min, z_max, 0.01)
    p_z_array = np.array([])
    for z_val in z_val_array:
        p_z_val = p_z(z_val)
        p_z_array = np.append(p_z_array, p_z_val)

    max_p_z = np.amax(p_z_array)

    # Sample a z value from the p(z) distribution
    z_val = 0
    for attempt in np.arange(0, 10):
        # Randomly generate many points
        batch = 10000
        z_guesses = rng.uniform(z_min, z_max, batch)
        y_guesses = rng.uniform(0, max_p_z, batch)

        # Accept those under the curve
        for i in np.arange(0, len(z_guesses)):
            if p_z(z_guesses[i]) >= y_guesses[i]:
                z_val = z_guesses[i]
                break
        if z_val > 0:
            break

    return z_val
