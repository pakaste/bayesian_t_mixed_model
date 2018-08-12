from numpy.random import chisquare
import numpy as np

from estimation.utils import calculate_covariance_u
from estimation.utils import calculate_sum
from estimation.utils import calculate_S_e
from settings import CONFIGS as cf


def estimate_s_e(sigma_e, nu_e, y, X, b, Z, u, family_indices):
    """
        Generates s_e for each i, where i = {1, ..., m} and m is the number of groups. In this case m is the number of families.
    """

    # Initialize s_e
    s_e = np.zeros(len(y))

    for ind in family_indices:
        df = nu_e + len(ind)
        X_copy = X.copy()
        b_copy = b.copy()
        Z_copy = Z.copy()
        u_copy = u.copy()

        # Calculate the S_e for every other observation, except belonging to this group
        X_subset = np.delete(X_copy, ind, axis=0)
        Z_subset = np.delete(Z_copy, ind, axis=0)
        b_subset = np.delete(b_copy, ind, axis=0)
        u_subset = np.delete(u_copy, ind, axis=0)

        print('X_subset ', X_subset.shape)
        print('b_subset ', b_subset.shape)
        print('Z_subset ', Z_subset.shape)
        print('u_subset ', u_subset.shape)

        S_e = calculate_S_e(sigma_e, nu_e, y, X_subset, b_subset, Z_subset, u_subset)

        # Calculate the estimate for the family i
        s_e_i = chisquare(df) / S_e

        try:
            s_e[ind[0]] = s_e_i
            s_e[ind[1]] = s_e_i
        except:
            s_e[ind[0]] = s_e_i

    return s_e


def estimate_s_u(Z, u, sigma_u, nu_u):
    """
        Generates s_u for each familial random effect.
    """
    df = nu_u + Z.shape[1]
    scaler = calculate_covariance_u(Z, u, sigma_u, nu_u)
    s_u_estimate = chisquare(df) / scaler

    return s_u_estimate


def estimate_sigma_e(s, y, X, b, Z, u, tau_e, Tau_e):
    S_t = calculate_sum(s, y, X, b, Z, u)
    nominator = (tau_e * Tau_e) + S_t
    df = Tau_e + X.shape[0]

    sigma_estimate = nominator / chisquare(df=df)

    return sigma_estimate