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
    Z = np.zeros((n, len(family_ids)))

    family_indices = []
    for i in range(len(family_ids)):
        idxs = X.index[X['family_nb'] == family_ids[i]].tolist()

        try:
            id1, id2 = idxs[0], idxs[1]
            Z[id1, i] = 1
            Z[id2, i] = 1
            family_indices.append([id1, id2])

        except IndexError as e:
            print('IndexError: Individuals matching family id: {}'.format(len(idxs)))
            id1 = idxs[0]
            Z[id1, i] = 1
            family_indices.append([id1])

    return Z, family_indices





