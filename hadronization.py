import numpy as np
import xarray as xr
import scipy.integrate as integrate
import timeit
import lhapdf

def frag(parton, num=1):
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
    z_val = np.array([])
    for attempt in np.arange(0, 10):
        # Randomly generate many points
        batch = 10000
        z_guesses = rng.uniform(z_min, z_max, batch)
        y_guesses = rng.uniform(0, max_p_z, batch)

        # Accept those under the curve
        for i in np.arange(0, len(z_guesses)):
            if p_z(z_guesses[i]) >= y_guesses[i]:
                z_val = np.append(z_val, z_guesses[i])
                if len(z_val) == num:
                    break
        if len(z_val) == num:
            break

    if len(z_val) == 1:
        return z_val[0]
    else:
        return z_val


# Function that takes an xarray dataarray and performs hard-soft coalescence on it using Boltzmann dist. thermal partons
def coal_xarray(xr_partons, T=0.155, max_pt=20):
    def soft_parts_boltz(E, phi=0, T=T, pid=21):
        if pid == 21:
            term = -1  # Bose statistics
            spin_factor = 3  # 2s + 1 for s = 1
            color_factor = 1
        elif np.abs(pid) <= 3:  # For light quarks
            term = 1  # Fermion statistics
            spin_factor = 2  # 2s + 1 for s = 1/2
            color_factor = 1
        else:  # Return zero -- don't involve these partons
            return 0

        return ((color_factor * spin_factor) / (np.exp(np.abs(E) / T) + term))

    # Get array bins and create a storage DataArray of zeros
    pt_array = xr_partons.pt.to_numpy()
    phi_array = xr_partons.phi.to_numpy()
    pid_array = xr_partons.pid.to_numpy()
    xr_hadrons = xr.DataArray(np.full((len(pt_array), len(phi_array)), float(0.0)),
                              coords={"pt": pt_array, "phi": phi_array})

    # Get pt & phi resolution from selected list, assuming it is constant
    pt_res = pt_array[-1] - pt_array[-2]
    phi_res = phi_array[-1] - phi_array[-2]

    # "momentum cutoffs in the phase space of quark-antiquark relative motions" inside pion
    delta_p = 0.24  # From https://arxiv.org/pdf/nucl-th/0301093 pg. 3

    # Iterate through dataarray cells, adding produced hadrons to new zeroes dataarray
    pt_i_arr = np.arange(0, len(xr_hadrons.pt))
    for pt_i in pt_i_arr[pt_i_arr <= max_pt]:  # Only perform coalescence up to maximum hadronic pt <max_pt>
        t_0 = timeit.default_timer()
        pt = pt_array[pt_i]
        print(pt)
        for phi_i in np.arange(0, len(xr_hadrons.phi)):

            phi = phi_array[phi_i]

            # Integrate over hard and soft pt for each flavor
            num_hadrons_dpt = 0
            for pt_hard in pt_array:
                for pid_val in pid_array:
                    # Get num hard particles in this pt and phi bin
                    num_hard = float(xr_partons.sel({"pt": pt_hard, "phi": phi, "pid": pid_val},
                                                    method='nearest').sum())  # Get arrary of hard particles in this phi bin
                    d_num_hard = num_hard / (pt_res * phi_res)

                    # Determine delta function constraints on soft pt
                    soft_min_delta = pt - (pt_res / 2) - pt_hard  # From momentum conserving delta function
                    soft_max_delta = pt + (pt_res / 2) - pt_hard

                    soft_min_hs = np.amin(
                        [(delta_p - pt_hard) / 2, (pt_hard - delta_p) / 2])  # From heavy side function
                    soft_max_hs = np.amax([(delta_p - pt_hard) / 2, (pt_hard - delta_p) / 2])

                    soft_min = np.amax([soft_min_delta, soft_min_hs, 0.01])  # Get most constraining window
                    soft_max = np.amin([soft_max_delta, soft_max_hs])

                    if soft_max <= soft_min or soft_max < 0.01:  # Integration returns zero, if bounds "twist"
                        continue

                    # Perform integral over soft sector and add to the count
                    quad_result = integrate.quad(lambda x: d_num_hard * soft_parts_boltz(x, phi, pid=pid_val) / phi_res,
                                                 soft_min, soft_max, points=[0])
                    num_hadrons_dpt += quad_result[0]

                num_hadrons = num_hadrons_dpt * pt_res  # integral from sum of bins
                num_hadrons = ((pt_res * phi_res) / (delta_p ** 3)) * num_hadrons  # Dimensionful prefactor

                xr_hadrons.loc[dict(pt=(pt), phi=(phi))] += num_hadrons  # Add produced hadrons to xarray

        t_f = timeit.default_timer()
        print(t_f - t_0)

    return xr_hadrons