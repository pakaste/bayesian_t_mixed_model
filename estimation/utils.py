import numpy as np


def initialize_parameters(n, cov_dim, initial_value, s_b, sigma_b, s_e, sigma_e, nu_e):
    # Initialize random coefficients for updating parameters
    coefficients = np.array([initial_value]*cov_dim)

    # Update variance parameters
    updated_s_b = np.zeros(n)
    updated_sigma_b = np.zeros(n)
    updated_s_e = np.zeros((n, len(s_e)))
    updated_sigma_e = np.zeros(n)
    updated_nu_e = np.zeros(n)

    # Set the first values calculated from hyperparameters
    updated_s_b[0] = s_b
    updated_sigma_b[0] = sigma_b

    updated_s_e[0, :] = s_e
    updated_sigma_e[0] = sigma_e

    updated_nu_e[0] = nu_e

    # Create empty array where to save the results
    estimates =  np.zeros(shape=(n, cov_dim))
    estimates[0, :] = coefficients

    return estimates, updated_s_b, updated_sigma_b, updated_s_e, updated_sigma_e, updated_nu_e


def calculate_uAu(Z, u):

    # Calculate covariance matrix for familial effects: A = Z^T Z
    Z_tranpose = np.transpose(Z)
    A = np.dot(Z_tranpose, Z)
    A_inv = np.linalg.inv(A)

    # Multiply by u
    u_Ainv = np.dot(np.transpose(u), A_inv)
    covariance = np.dot(u_Ainv, u)

    return covariance


def calculate_covariance_u(Z, u, sigma_u, nu_u):
    uAu = calculate_uAu(Z, u)
    scaled_S_u = (uAu / sigma_u) + nu_u

    return scaled_S_u


def calculate_sse(y, X, b, Z, u):

    Xb = np.dot(X, b)
    Zu = np.dot(Z, u)

    # SSE^T SSE
    sse = y - Xb - Zu
    sse_transpose = np.transpose(sse)

    # Take the matrix multiplication
    dot_product = np.dot(sse_transpose, sse)

    return dot_product


def calculate_S_e(sigma_e, nu_e, y, X, b, Z, u):

    sse = calculate_sse(y, X, b, Z, u)
    scaled_sse = (sse / sigma_e) + nu_e

    if np.isfinite(scaled_sse):
        return scaled_sse
    else:
        print('SSE not finite:')
        print('sse ', sse)
        print('sigma_e: ', sigma_e)
        print('nu_e: ', nu_e)

        return None


def calculate_sum(s, y, X, b, Z, u):
    """
        Parameters:
            s   : (1d array) of size (n, )
            y   : (1d array) of size (n, )
            X   : (2d array) of size (n, p)
            b   : (1d array) of size (p, )
            Z   : (2d array) of size (n, q)
            u   : (1d array) of size (q, )
        Returns:
            (int)
    """
    dot_product = calculate_sse(y, X, b, Z, u)
    S_t_m = s * dot_product

    return S_t_m