from numpy.random import randint
import warnings
import numpy as np


def divide_into_training_and_testing(X, y, family_id):
    """
        Divides data into training and testing according to the family id. Members from the same family will always belong to either training or testing, not different sets.
    """

    family_id = list(set(family_id))
    family_train_idx = randint(0, len(family_id), size=450)

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


def create_familial_matrix(X, family_ids):
    n = X.shape[0]
    Z = np.zeros((n, n))

    for famid in family_ids:
        idxs = X.index[X['family_nb'] == famid].tolist()

        try:
            id1, id2 = idxs[0], idxs[1]
            Z[id1, id1] = 1
            Z[id2, id2] = 1
            Z[id1, id2] = 0.98 # np.corr(y[id1], y[id2])
            Z[id2, id1] = 0.98
        except IndexError as e:
            print('IndexError: Individuals matching family id: {}'.format(len(idxs)))
            id1 = idxs[0]
            Z[id1] = 1

    # Remove all zero rows from Z
    Z[~np.all(Z == 0, axis=1)]

    return Z





