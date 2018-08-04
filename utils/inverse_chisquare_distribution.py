import numpy as np


def inverse_chisquare(df, scale, size=1, scaled_inverse=False):
    """Generates numbers from inverse-chisquare distribution with parameters df and 1/df"""

    if scaled_inverse:
        return ((df * scale) / np.random.chisquare(df=df, size=size))[0]
    else:
        return 1 / np.random.chisquare(df=df, size=size)[0]

