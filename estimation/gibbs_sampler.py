"""
    Functions to estimate the best linear unbiased predictors for
    coefficient vector u as stated in Strandén & Gianola.
"""

import sys
from datetime import datetime
import numpy as np
from numpy.random import normal

from estimation.hendersson_equations import henderson_model_equations
from estimation.variance_parameters import estimate_s_e
from estimation.variance_parameters import estimate_s_u
from estimation.variance_parameters import estimate_sigma_e
from estimation.variance_parameters import estimate_sigma_u
from estimation.degrees_of_freedom import discrete_nu_e
from estimation.utils import initialize_parameters


# Initialize individual error terms

def run_gibbs_sampler(y, X, Z, s_b, sigma_b, tau_b, Tau_b, nu_b, s_e, sigma_e, tau_e, Tau_e, nu_e, family_indices, initial_value, n):

    """
        Updates the BLUP parameters as well as variance components for for familial and individual terms with MCMC sampling scheme. Tau and tau values have to given as hyperparameters.
    """

    # Initialize random coefficients for updating parameters
    p_dim = X.shape[1]
    q_dim = Z.shape[1]
    cov_dim = p_dim + q_dim

    estimates, updated_s_b, updated_sigma_b, updated_s_e, updated_sigma_e, updated_nu_e = initialize_parameters(n, cov_dim, initial_value, s_b, sigma_b, s_e, sigma_e, nu_e)

    # Start updating the parameters
    for i in range(1, estimates.shape[0]):

        # Update the BLUP parameters
        # Calculate updated Henderson model equation
        coefficient_matrix, right_hand = henderson_model_equations(y, X, Z, updated_s_b[i-1], updated_sigma_b[i-1], updated_s_e[i-1], updated_sigma_e[i-1])

        # Update coefficients
        for j in range(cov_dim):
            sum1 = 0
            sum2 = 0

            if j > 0:
                sum1 = np.dot(coefficient_matrix[j, :j], estimates[i, :j])
            if j < cov_dim-1:
                sum2 = np.dot(coefficient_matrix[j, (j+1):], estimates[i-1, (j+1):])

            final_sum = sum1 + sum2

            # Generate mean value from Normal distribution
            a_worm_i = (1/coefficient_matrix[j, j]) * (right_hand[j] - final_sum)
            scale = np.sqrt((1/coefficient_matrix[j, j]))

            estimates[i, j] = np.random.normal(loc=a_worm_i, scale=scale)

        if i % 1000 == 0:
            print('{}th iteration finished'.format(i))

        # Update s_e_i
        updated_s_e[i, :] = estimate_s_e(updated_sigma_e[i-1], nu_e, y, X, estimates[i, :p_dim], Z, estimates[i, p_dim:], family_indices)

        # Update s_u
        updated_s_b[i] = estimate_s_u(Z, estimates[i, p_dim:], updated_sigma_b[i-1], nu_b)

        # Update sigma_e
        updated_sigma_e[i] = estimate_sigma_e(updated_s_e[i-1, :], y, X, estimates[i, :p_dim], Z, estimates[i, p_dim:], tau_e, Tau_e, family_indices)

        # Update Sigma_b
        updated_sigma_b[i] = estimate_sigma_u(updated_s_b[i-1], y, X, estimates[i, :p_dim], Z, estimates[i, p_dim:], tau_b, Tau_b)

        # Update nu_e
        #updated_nu_e[i] = discrete_nu_e(nu_e, s_e, len(family_indices))

    return estimates, updated_s_e, updated_s_b, updated_sigma_e, updated_sigma_b, updated_nu_e


def run_one_chain(params, iters):

    y = params[0]
    X = params[1]
    Z = params[2]
    s_b = params[3]
    sigma_b = params[4]
    tau_b = params[5]
    Tau_b = params[6]
    nu_b = params[7]
    s_e = params[8]
    sigma_e = params[9]
    tau_e = params[10]
    Tau_e = params[11]
    nu_e = params[12]
    family_indices = params[13]

    # Randomly start initial value
    initial_value = abs(0.1*normal(loc=0.0, scale=1.0))

    # Run gibbs sampler
    final_estimates, updated_s_e, updated_s_u, updated_sigma_e, updated_sigma_b, updated_nu_e = run_gibbs_sampler(y, X, Z, s_b, sigma_b, tau_b, Tau_b, nu_b, s_e, sigma_e, tau_e, Tau_e, nu_e, family_indices, initial_value=initial_value, n=iters)

    bayes_estimates = [final_estimates, updated_s_e, updated_s_u, updated_sigma_e, updated_sigma_b, updated_nu_e]

    return bayes_estimates

