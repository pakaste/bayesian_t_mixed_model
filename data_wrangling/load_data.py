import warnings
import pandas as pd
import numpy as np
from random import randint
from sklearn.preprocessing import OneHotEncoder
import warnings

from settings import CONFIGS as cf


def load_data(path='', all_cpgs=False):
    """
        Loads CpG and phenotype data from a given path and returns two pandas dataframes.
    """
    cpg = pd.read_csv(path + '/estimated_top_cpgs_age.csv')
    pheno = pd.read_csv(path + '/pheno_data_MZ.csv')

    # If all CpGs aren't imported, pick randomly one CpG
    if not all_cpgs:
        rand_int = randint(0, cpg.shape[0])
        cpg = cpg.iloc[rand_int]

    cpg_transpose = cpg.transpose()

    # Order the data according to the barcode
    pheno.sort_values(['Barcode'], ascending=True, inplace=True)
    pheno.reset_index(inplace=True, drop=True)

    cpg_transpose.sort_index(ascending=True, inplace=True)

    # Check that both dataframes are in the same order:
    if not (cpg_transpose.index.values == pheno['Barcode'].values).all():
        warnings.warn('Data is not in the correct order')

    return cpg_transpose, pheno


def subset_pheno(pheno_data, remove_nans=False, categorize=False, remove_correlated=False):
    """
        Subsets phenotype data according to the covariates and convert it to the numpy array
    """
    subset = pheno_data[cf.covariates]

    if categorize:
        # Convert gender to 'FEMALE' = 0, 'MALE' = 1
        gender_binary = [1 if gndr is 'MALE' else 0 for gndr in subset['GENDER']]
        subset['GENDER'] = gender_binary

        # Convert smoking into categorical one
        subset.loc[:, 'smoking'] = [int(x) if x != '2 or 1' else '3' for x in subset['smoking'].values]
        subset = pd.get_dummies(subset, columns=['smoking'])

        # Drop one column for reference
        subset = subset.drop(columns=['smoking_0'])

    if remove_nans:
        subset = subset.dropna()

    if remove_correlated:
        subset = subset.drop(columns=['CD4T', 'PC2_cp', 'PC5_cp'])

    return subset