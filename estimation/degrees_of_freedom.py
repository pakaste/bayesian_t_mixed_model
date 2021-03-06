import numpy as np
import warnings
from scipy.special import gamma


def calculate_P_e(s_e, m):
    """
        Calculates P_e for updating the nu_e.
        Parameters:
            s_e     : (m, ) numpy array of familial mixing effect
            m       : (int) number of groups
        Returns:
            (int)
    """

    multiplier = s_e[0]
    for i in range(1, len(s_e)):
        multiplier *= (s_e[i]**2)

    pe = multiplier**(1/m)

    return pe


def calculate_fj(d_e):
    """
        Calculates the prior probability for nu_e.
        Parameters:
            (int)
        Returns:
            float
    """

    return d_e


def calculate_C_e(s_e, m, d_e):
    """
        Calculates the C_e given familial mixing parameter and beliefs d_e for nu_e prior.
        Parameters:
            s_e     : (m, ) numpy array of familial mixing process
            m       : (int) number of groups
            d_e     : (list) of hyperparameters of residual term degrees of freedoms
        Returns:
            (float)
    """

    sum_all = 0
    P_e = calculate_P_e(s_e, m)

    # Calculate the sum over m groups of s_e^2
    s_e_sum2 = np.sum(s_e**2)

    for j in range(len(d_e)):
        print('\nj: ', j)

        fj = calculate_fj(d_e[j])

        # Calculate parts
        multiplier = (m*fj)/2
        print('multiplier: ', multiplier)
        gamma_term = (gamma(fj/2))**(-m)
        print('gamma_term: ', gamma_term)
        exp_term = np.exp(-(fj/2)*s_e_sum2)
        print('exp_term: ', exp_term)

        # The j:th sum
        sum_j = (((fj * P_e)/2)**multiplier) * gamma_term * exp_term
        print('sum_j: ', sum_j)

        sum_all = sum_all + sum_j

    return sum_all


def discrete_nu_e(nu_e, s_e, m, d_e=[2, 4, 10, 100, 1000]):
    """
        Updates the degrees of freedom parameter nu_e if prior distribution for nu_e is discrete, instead of continuous.
    """

    # Calculate P_e
    P_e = calculate_P_e(s_e, m)

    if P_e == None:
        warnings.warn('P_e is None!')
        return None

    inv_C_e = calculate_C_e(s_e, m, d_e)

    if inv_C_e == 0:
        warnings.warn('C_e is exactly zero!')

    # Calculate the sum over m groups of s_e^2
    s_e_sum2 = np.sum(s_e**2)
    exp_term = np.exp(-(nu_e/2) * s_e_sum2)

    gamma_term = (gamma(nu_e/2))**(-m)

    multiplier = (m*nu_e)/2
    updated_nu = C_e * ((nu_e*P_e/2)**multiplier) * gamma_term * exp_term

    return updated_nu
