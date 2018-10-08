from numpy.random import randint
import warnings
import numpy as np
import pandas as pd

from settings import CONFIGS as cf


def divide_into_training_and_testing(X, y, family_id):
    """
        Divides data into training and testing according to the family id. Members from the same family will always belong to either training or testing, not different sets.
    """

    family_id = list(set(family_id))
    family_train_idx = randint(0, len(family_id), size=cf.TRAIN_SET)

    train_family_id = [family_id[i] for i in family_train_idx]
    mask_train = X['family_nb'].isin(train_family_id)

    # Take training set
    X_train = X[mask_train]
    y_train = y[mask_train]

    # Take testing set
    X_test = X[~mask_train]
    y_test = y[~mask_train]

    # Reset the indices
    X_train.reset_index(inplace=True, drop=True)
    X_test.reset_index(inplace=True, drop=True)
    y_train.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)

    return X_train, y_train, X_test, y_test


def create_Z_matrix(data):
    Z_matrix = pd.get_dummies(data['family_nb'])

    # Check that columns match those in the data
    z_cols = set(Z_matrix.columns)
    data_categories = set(data['family_nb'])

    if z_cols != data_categories:
        print('Z matrix columns does not match data categories!')
        return None

    # Check how many columns had only one individual
    col_sums = Z_matrix.sum(axis=0)
    mask = (col_sums < 2)

    # Create family indices list
    family_indices = []
    for family_ind in Z_matrix.columns:
        idxs = data.index[data['family_nb'] == family_ind].tolist()
        if len(idxs) == 2:
            family_indices.append([idxs[0], idxs[1]])
        else:
            family_indices.append([idxs[0]])

    return np.array(Z_matrix), family_indices
