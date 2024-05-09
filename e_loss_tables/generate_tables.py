import numpy as np
import scipy as sp
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import logging
import time

subdiv = 1
for parton in ['q', 'g']:
    print('Computing tables for parton: {}'.format(parton))
    for coupling in [1.8, 1.9, 2.0, 2.1, 2.2]:
        print('Computing tables for g={}'.format(coupling))
        ALPHAS = (coupling ** 2) / (4 * np.pi)


        ############################
        # Define medium parameters #
        ############################

        # Method to return partial density at a particular point for given medium partons
        # Chosen to be ideal gluon gas dens. as per Sievert, Yoon, et. al.
        def rho(T, med_parton='g'):
            if med_parton == 'g':
                density = 1.202056903159594 * 16 * (1 / (np.pi ** 2)) * (T ** 3)
            elif med_parton == 'q':
                density = 1.202056903159594 * (3 / 4) * 24 * (1 / (np.pi ** 2)) * (T ** 3)
            else:
                # Return 0
                density = 0
            return density


        # Function to return total cross section at a particular point for parton and *gluon* in medium
        # Total GW cross section, as per Sievert, Yoon, et. al.
        # Specify med_parton either 'g' for medium gluon or 'q' for generic light (?) quark in medium
        # https://inspirehep.net/literature/1725162
        def sigma(T, parton, med_parton='g'):
            """
            We select the appropriate cross-section for a known parton and
            known medium parton specified when called
            """
            # current_point = point

            parton_type = parton

            sigma_gg_gg = (9 / (32 * np.pi)) * coupling ** 4 / ((coupling * T) ** 2)
            sigma_qg_qg = (1 / (8 * np.pi)) * coupling ** 4 / ((coupling * T) ** 2)
            sigma_qq_qq = (1 / (18 * np.pi)) * coupling ** 4 / ((coupling * T) ** 2)

            if parton_type == 'g' and med_parton == 'g':
                # gg -> gg cross-section
                cross_section = sigma_gg_gg
            elif parton_type == 'q' and med_parton == 'g':
                # qg -> qg cross-section
                cross_section = sigma_qg_qg
            elif parton_type == 'g' and med_parton == 'q':
                # qg -> qg cross-section
                cross_section = sigma_qg_qg
            elif parton_type == 'q' and med_parton == 'q':
                # qq -> qq cross-section
                cross_section = sigma_qq_qq
            else:
                logging.debug('Unknown parton scattering cs... Using gg->gg scattering cross section')
                cross_section = sigma_gg_gg

            return cross_section


        # Function to return inverse QGP drift mean free path in units of GeV^{-1}
        # Total GW cross section, as per Sievert, Yoon, et. al.
        def inv_lambda(T, parton='q', med_parton='all'):
            """
            We apply a reciprocal summation between the cross-section times density for a medium gluon and for a medium quark
            to get the mean free path as in https://inspirehep.net/literature/1725162
            """

            if med_parton == 'all':
                return (sigma(T, parton, med_parton='g') * rho(T, med_parton='g')
                        + sigma(T, parton, med_parton='q') * rho(T, med_parton='q'))
            else:
                return sigma(T, parton, med_parton=med_parton) * rho(T, med_parton=med_parton)


        ####################
        # Define integrand #
        ####################

        # Create interpolated function for delta E given T, L, & E:
        def delta_E(E, T, L):

            if parton == 'q':
                CR = 4 / 3
                # L = 3  # in fm
                mu = coupling * T  # in GeV, for g * T
                # E = 50  # jet E in GeV
                lamb = 1 / inv_lambda(mu / 2, parton=parton)  # in GeV
            else:
                CR = 3
                # L = 3  # in fm
                mu = coupling * T  # in GeV, for g * T
                # E = 50  # jet E in GeV
                lamb = 1 / inv_lambda(mu / 2, parton=parton)  # in GeV
            FmGeV = 1 / 0.19732687

            # # Define the analytic dI_dx we're targeting
            # def an_dI_dx(x):
            #     return ( ((1 / FmGeV) ** 2) * (CR * ALPHAS / 4) * ((1 - x + ((x**2)/2))/x) * ((L**2 * mu**2)/lamb))

            # Define the analytic Delta E at first order
            an_delta_E_1 = (((FmGeV) ** 2) * (CR * ALPHAS / 4) * ((L ** 2 * mu ** 2) / lamb) * np.log(E / mu))

            # Define analytic Delta E at zeroth order (vacuum)
            an_delta_E_0 = ((4 * CR * ALPHAS / (3 * np.pi)) * E * np.log(E / mu))

            an_delta_E = an_delta_E_1  # an_delta_E_0 + an_delta_E_1

            # Define numerical integrand as function of q and k

            # Define numerical integrand as function of q and k
            def integrand(x):
                return lambda phi, k, q: (((FmGeV) ** 3) * (4 * CR * ALPHAS / (np.pi ** 2))
                                          * (1 - x + ((x ** 2) / 2))
                                          * (L / lamb) * E
                                          * ((mu ** 2) / ((q ** 2 + mu ** 2) ** 2))
                                          * ((q ** 2 * np.cos(phi) * (
                                    k ** 2 - 2 * k * q * np.cos(phi) + q ** 2) * L ** 2)
                                             / (16 * x ** 2 * E ** 2 + (
                                        (k ** 2 - 2 * k * q * np.cos(phi) + q ** 2) ** 2 * L ** 2 * ((FmGeV) ** 2)))))

            abs_err = 0.1
            rel_err = 0.1

            dI_dx_finq = lambda x: 2 * (integrate.nquad(integrand(x), [[0, np.pi], [mu, np.min(
                [2 * E * x, 2 * E * np.sqrt(x * (1 - x))])], [0, np.sqrt(3 * mu * E)]],
                                                        opts={"epsabs": abs_err, "epsrel": rel_err, "limit": subdiv})[
                0])

            x_min = 0
            x_max = 1
            t0 = time.time()
            delta_E = integrate.quad(dI_dx_finq, x_min, x_max, limit=subdiv)[0]
            tf = time.time()
            print('E={} GeV, T={} GeV, L={} fm -- deltaT={} s'.format(E, T, L, (tf - t0)))

            return delta_E


        delta_E = np.vectorize(delta_E)  # IMPORTANT!!!

        ##############################
        # Sample Delta E Phase Space #
        ##############################

        E_points = np.logspace(0, 2, 10)  # Logarithmic in 1 to 100 GeV
        T_points = np.logspace(-0.826814, -0.154902,
                               10)  # Logarithmic in 0.149 to 0.7 -- Seen in datasets as range of Tmax_event
        L_points = np.logspace(-0.6020599913279624, 1.4,
                               12)  # Logarithmic in 0.25 to 25 -- Seen in datasets range of time_total_plasma is 0.37-15, but we extend for gradients
        #  g_points = np.array([1.8, 1.9, 2, 2.1, 2.2])

        Es, Ts, Ls = np.meshgrid(E_points, T_points, L_points, indexing='ij')

        delta_E_vals = delta_E(Es, Ts, Ls)

        np.savez('g{}_deltaE_samples_{}_{}subdiv.npz'.format(coupling, parton, subdiv), E_points=E_points,
                 T_points=T_points,
                 L_points=L_points, delta_E_vals=delta_E_vals)
