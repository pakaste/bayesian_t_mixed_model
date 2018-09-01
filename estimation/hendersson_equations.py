import numpy as np
from numpy.linalg import inv

"""
    Henderson's mixed model equations for estimating the model parameters.
"""

def henderson_model_equations(y, X, Z, s_b, sigma_b, s_e, sigma_e):

    # Check that s_e is the same length as Z.shape[1]
    if len(s_e) != len(y):
        print('Length s_e != len(y)')
        return None

    diagonal_value = np.multiply(s_e, (1 / sigma_e))
    R_inv = np.diag(diagonal_value)

    # Create transpose of matrices
    transpose_X = np.transpose(X)
    transpose_Z = np.transpose(Z)

    # Upper part of the mixed model equations
    uLeft = np.dot(np.dot(transpose_X, R_inv), X)
    uRight = np.dot(np.dot(transpose_X, R_inv), Z)
    upper_matrix = np.hstack((uLeft, uRight))

    # Lower part of the mixed model equations
    lLeft = np.dot(np.dot(transpose_Z, R_inv), X)

    # Variance-covariance matrix of additive relationships:
    inverse_A = inv(np.dot(transpose_Z, Z))

    # Lower, right part of model equations
    lRight = np.dot(np.dot(transpose_Z, R_inv), Z) + np.multiply((s_b / sigma_b), inverse_A)
    lower_matrix = np.hstack((lLeft, lRight))

    full_matrix = np.vstack((upper_matrix, lower_matrix))

    # Right hand side of the equations
    upper = np.dot(np.dot(transpose_X, R_inv), y)
    lower = np.dot(np.dot(transpose_Z, R_inv), y)
    right_hand = np.concatenate((upper, lower))

    # Check that left hand of equations dimension match with right hand
    if full_matrix.shape[1] != right_hand.shape[0]:
        raise TypeError('Dimensions of left and right hand equations do not match!')

    return full_matrix, right_hand

