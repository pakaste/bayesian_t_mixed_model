from numpy.random import chisquare
import numpy as np

from estimation.utils import calculate_uAu
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
        y_copy = np.squeeze(y.copy())
        X_copy = X.copy()
        Z_copy = Z.copy()

        # Calculate the S_e for every other observation, except belonging to this group
        y_subset = np.delete(y_copy, ind, axis=0)
        X_subset = np.delete(X_copy, ind, axis=0)
        Z_subset = np.delete(Z_copy, ind, axis=0)

        S_e = calculate_S_e(sigma_e, nu_e, y_subset, X_subset, b, Z_subset, u)

        # Calculate the estimate for the family i
        s_e_i = chisquare(df) / S_e

        # Update the s_e for ith family
        s_e[ind] = s_e_i

    return s_e


def estimate_s_u(Z, u, sigma_u, nu_u):
    """
        Generates s_u for each familial random effect.
    """
    df = nu_u + Z.shape[1]
    scaler = calculate_covariance_u(Z.copy(), u.copy(), sigma_u, nu_u)
    s_u_estimate = chisquare(df) / scaler

    return s_u_estimate


def estimate_sigma_e(s_e, y, X, b, Z, u, tau_e, Tau_e, family_indices):
    """
        Updates the estimate for sigma_e
    """

    S_t = 0
    y_copy = y.copy()
    X_copy = X.copy()
    Z_copy = Z.copy()

    for family_ind in family_indices:

        # Calculate the S_t for every group m separately
        y_subset = np.squeeze(y_copy[family_ind])
        X_subset = X_copy[family_ind]
        Z_subset = Z_copy[family_ind]
        s_e_i = s_e[family_ind][0]

        group_m_S_t = calculate_sum(s_e_i, y_subset, X_subset, b, Z_subset, u)
        S_t = S_t + group_m_S_t

    nominator = (tau_e * Tau_e) + S_t
    df = tau_e + X.shape[0]

    sigma_estimate = nominator / chisquare(df=df)

    return sigma_estimate


def estimate_sigma_u(s_u, y, X, b, Z, u, tau_u, Tau_u):
    covariance_u = calculate_uAu(Z.copy(), u.copy())
    nominator = (tau_u*Tau_u + s_u*covariance_u)

    df = tau_u + Z.shape[1]
    updated_sigma_u = nominator / chisquare(df=df)

    return updated_sigma_u