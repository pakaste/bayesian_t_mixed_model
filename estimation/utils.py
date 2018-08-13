import numpy as np


def calculate_uAu(Z, u):
    Z_tranpose = np.transpose(Z)
    A = np.dot(Z_tranpose, Z)
    A_inv = np.linalg.inv(A)

    # Multiply by u
    covariance = np.dot(np.dot(np.transpose(u), A_inv), u)

    return covariance


def calculate_covariance_u(Z, u, sigma_u, nu_u):
    S_u = calculate_uAu(Z, u)
    scaled_S_u = (S_u / (1 / sigma_u)) + nu_u

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

    return scaled_sse


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