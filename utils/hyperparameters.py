import numpy as np
from utils.inverse_chisquare_distribution import inverse_chisquare


def initialize_individual_parameters(n, tau_e=1.0, Tau_e=0.01, nu_e=2.0):
    """
        Function to initialize individual parameters based on hyperparameters tau_e, Tau_e and degrees of freedom nu_e

        Parameters:
            n     : (int) amount of observations
            tau_b : (float)
            Tau_b : (float)
            nu_b  : (float) degrees of freedom
        Returns: sigma_e, s_e
    """

    # The inpendent individual error term
    sigma_e = inverse_chisquare(df=tau_e, scale=Tau_e, size=1, scaled_inverse=True)

    s_e = np.zeros(shape=n)
    i = 0
    while (i <= n):
        mix_param = np.random.chisquare(df=nu_e, size=1) / nu_e
        s_e[i:(i + 2)] = mix_param
        i = i + 2

    scale_param = sigma_e / s_e

    return sigma_e, s_e, scale_param


def initialize_familial_parameters(n_fam, tau_b=2.0, Tau_b=0.05, nu_b=3.0):
    """
        Function to initialize famialial parameters based on hyperparameters tau_b, Tau_b and degrees of freedom nu_b

        Parameters:
            n_fam : (int) number of different families
            tau_b : (float)
            Tau_b : (float)
            nu_b  : (float) degrees of freedom
        Returns: sigma_b, s_b, b
    """
    sigma_b = inverse_chisquare(df=tau_b, scale=Tau_b, size=1, scaled_inverse=True)

    # Familial hyperparameter
    s_b = np.random.chisquare(df=nu_b, size=1) / nu_b
    b = np.random.normal(loc=0.0, scale=(sigma_b / s_b), size=n_fam)

    return sigma_b, s_b, b