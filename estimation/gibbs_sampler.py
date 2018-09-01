"""
    Functions to estimate the best linear unbiased predictors for
    coefficient vector u as stated in Strandén & Gianola.
"""

import numpy as np

from estimation.hendersson_equations import henderson_model_equations
from estimation.variance_parameters import estimate_s_e
from estimation.variance_parameters import estimate_s_u
from estimation.variance_parameters import estimate_sigma_e
from estimation.variance_parameters import estimate_sigma_u
from estimation.utils import initialize_parameters


# Initialize individual error terms

def run_gibbs_sampler(y, X, Z, s_b, sigma_b, tau_b, Tau_b, nu_b, s_e, sigma_e, tau_e, Tau_e, nu_e, family_indices, initial_value=0.0001, n=1000):

    """
        Updates the BLUP parameters as well as variance components for for familial and individual terms with MCMC sampling scheme. Tau and tau values have to given as hyperparameters.
    """

    # Initialize random coefficients for updating parameters
    p_dim = X.shape[1]
    q_dim = Z.shape[1]
    cov_dim = p_dim + q_dim

    estimates, updated_s_b, updated_sigma_b, updated_s_e, updated_sigma_e =initialize_parameters(n, cov_dim, initial_value, s_b, sigma_b, s_e, sigma_e)

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
        #print('\nupdated_s_b[i]: ', updated_s_b[i])

        # Update sigma_e
        updated_sigma_e[i] = estimate_sigma_e(updated_s_e[i-1, :], y, X, estimates[i, :p_dim], Z, estimates[i, p_dim:], tau_e, Tau_e, family_indices)
        #print('updated_sigma_e[i]: ', updated_sigma_e[i])

        # Update Sigma_b
        updated_sigma_b[i] = estimate_sigma_u(updated_s_b[i-1], y, X, estimates[i, :p_dim], Z, estimates[i, p_dim:], tau_b, Tau_b)
        #print('updated_sigma_b[i]: ', updated_sigma_b[i])


    return estimates, updated_s_e, updated_s_b, updated_sigma_e, updated_sigma_b