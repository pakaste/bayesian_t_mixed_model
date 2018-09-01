import numpy as np
from estimation.hendersson_equations import henderson_model_equations


def update_BLUP_estimates(y, X, Z, s_b, sigma_b, s_e, sigma_e, estimates_i,estimates_prev, cov_dim):
    """
        Updates the BLUP (Best Linear Unbiased Parameters) parameters using Henderson model equations and sampling the BLUP estimates from Normal distribution.
        Parameters:
            y               : (n, ) numpy array of dependent variable
            X               : (n, p) numpy array of independent variables
            Z               : (n, q) numpy array of incidence matrix for familial effects
            s_b             : (n, ) numpy array of mixing process of familial terms
            sigma_b         : (n, ) numpy array of variance of familial effects
            s_e             : (n, ) numpy array of mixing process of individual effects
            sigma_e         : (n, ) numpy array of variance of individual effects
            estimates_i     : (1, cov_dim) numpy array of current MCMC BLUP estimates
            estimates_prev  : (1, cov_dim) numpy array of MCMC BLUP estimates of the previous row
            cov_dim         : (int) dimensions of X and Z matrices
        Returns:
            (list) estimated BLUP parameters
    """

    # Calculate updated Henderson model equation
    coefficient_matrix, right_hand = henderson_model_equations(y=y, X=X, Z=X, s_b=s_b, sigma_b=sigma_b, s_e=s_e, sigma_e=sigma_e)

    # Update coefficients
    blup_estimates = []
    for j in range(cov_dim):
        sum1 = 0
        sum2 = 0

        # Check whether the j:th variable is the very first or the very last
        if j > 0:
            sum1 = np.dot(coefficient_matrix[j, :j], estimates_i[:j])
        if j < cov_dim-1:
            sum2 = np.dot(coefficient_matrix[j, (j+1):], estimates_prev[(j+1):])
        final_sum = sum1 + sum2

        # Generate the update parameter value from Normal distribution
        a_worm_i = (1/coefficient_matrix[j, j]) * (right_hand[j] - final_sum)
        scale = np.sqrt((1/coefficient_matrix[j, j]))
        estimate_a = np.random.normal(loc=a_worm_i, scale=scale)

        # Append the estimates BLUPS to a list
        blup_estimates.append(estimate_a)

    return blup_estimates


