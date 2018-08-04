import pandas as pd
import numpy as np
from random import randint
from sklearn.preprocessing import OneHotEncoder

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

    return cpg_transpose, pheno


def subset_pheno(pheno_data, remove_nans=False, categorize=False):
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

    if remove_nans:
        subset = subset.dropna()

    return np.array(subset)