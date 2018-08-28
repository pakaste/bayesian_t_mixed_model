import random
import numpy as np
from numpy.random import normal
from sklearn.preprocessing import OneHotEncoder

from utils.hyperparameters import initialize_familial_parameters
from utils.hyperparameters import initialize_individual_parameters


def simulate_fake_data(n, p, tau_b, Tau_b, nu_b, tau_e, Tau_e, nu_e, seed=10):

    # Set the PRNG seed
    random.seed(seed)

    # Generate simulated X matrix
    X = generate_data(n, p)
    print('Shape simulated of X: ', X.shape)

    # Generate simulated beta
    beta = generate_parameters(p=p)
    print('Shape simulated of beta: ', beta.shape)

    # Create simulated Z matrix
    n_families = int(n/2)
    Z = np.zeros(shape=(n, n_families))
    print('Shape simulated of Z', Z.shape)

    # Fill Z matrix
    j = 0
    family_indices = []
    for i in range(n_families):
        family_indices.append([j, j+1])
        Z[j:(j + 2), i] = np.array([1, 1])
        j = j + 2

    print('Initializing hyperparameters')

    # Familial effects
    sigma_b, s_b, b = initialize_familial_parameters(n_fam=n_families, tau_b=tau_b, Tau_b=Tau_b, nu_b=nu_b)

    # The inpendent individual error term
    sigma_e, s_e, scale_param = initialize_individual_parameters(family_indices=family_indices, n=n, tau_e=tau_e, Tau_e=Tau_e, nu_e=nu_e)

    print('scale_param:' , scale_param)

    print('Initializing dependent variable')
    y = np.dot(X, beta) + np.dot(Z, b) + np.random.normal(loc=0, scale=scale_param, size=n)

    return y, X, beta, Z, sigma_b, s_b, b, sigma_e, s_e, scale_param, family_indices


def generate_data(n=100, p=10, categorical=False):
    """
        Generates artificial data.

        Parameters:
            n             : (int) amount of observations (rows)
            p             : (int) amount of variables (columns)
            categorical   : (bool) whether to include categorical variables
        Returns: (n x p) numpy array
    """

    df = np.zeros((n, p))
    start_col = 1

    if categorical:
        cat_var = int(np.random.uniform(low=0, high=4, size=n))
        encoder = OneHotEncoder()
        categorical_variable = encoder.fit_transform(cat_var)
        df[:, :4] = categorical_variable
        start_col = 4

    df[:, 0] = np.ones(n)

    for col in range(start_col, p):
        mean = np.random.uniform(low=-20, high=50, size=1)
        scale = np.sqrt(np.abs(np.random.normal(loc=1.0, scale=3)))
        df[:, col] = np.random.normal(loc=mean, scale=scale, size=n)

    return df


def generate_parameters(intercept=-20, p=10):
    params = np.zeros(p)
    params[0] = intercept

    for i in range(1, p):
        params[i] = np.random.normal(loc=0, scale=1.0, size=1)

    return params





