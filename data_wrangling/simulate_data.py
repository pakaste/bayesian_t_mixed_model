import numpy as np
from sklearn.preprocessing import OneHotEncoder


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





