"""
    Functions to estimate the best linear unbiased predictors for
    coefficient vector u as stated in Strandén & Gianola.
"""

import numpy as np


# Initialize individual error terms

def estimate_BLUP(coefficient_matrix, coefficients, right_hand, n=1000):
    ncol = len(coefficients)
    print('ncol:', ncol)
    print('shape of coefficients matrix ', coefficient_matrix.shape)
    estimates =  np.zeros(shape=(n, ncol))
    estimates[0, :] = coefficients
    print('estimates ', estimates.shape)

    for i in range(1, estimates.shape[0]):
        for j in range(ncol):
            sum1 = 0
            sum2 = 0

            if j > 0:
                sum1 = np.dot(coefficient_matrix[j, :j],
                    estimates[i, :j])
            if j < ncol-1:
                sum2 = np.dot(coefficient_matrix[j, (j+1):],
                    estimates[i-1, (j+1):])

            final_sum = sum1 + sum2
            a_worm_i = (1/coefficient_matrix[j, j]) * (right_hand[j] - final_sum)
            scale = np.sqrt((1/coefficient_matrix[j, j]))

            estimates[i, j] = np.random.normal(loc=a_worm_i, scale=scale)

        if i % 1000 == 0:
            print('{}th iteration finished'.format(i))

    return estimates