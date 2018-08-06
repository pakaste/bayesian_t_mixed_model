import os
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

from data_wrangling.load_data import load_data, subset_pheno
from data_wrangling.training_testing_set import divide_into_training_and_testing
from data_wrangling.training_testing_set import create_familial_matrix
from visualization.plotting import plot_histogram
from utils.hyperparameters import initialize_familial_parameters
from utils.hyperparameters import initialize_individual_parameters
from estimation.hendersson_equations import henderson_model_equations
from estimation.blup_estimates import estimate_BLUP
from settings import CONFIGS as cf


# Read in data
print('Working directory: ', os.getcwd())
cpg, pheno = load_data('data', all_cpgs=False)
cpg = pd.DataFrame(cpg)

#
individual_id = cpg.index
cpg.reset_index(inplace=True, drop=True)
print(cpg.shape)

print('Shape of cpg data: ', cpg.shape)
print('Shape of pheno data:', pheno.shape)


# Subset data
pheno_subset = subset_pheno(pheno, categorize=True, remove_correlated=True)
print('Shape of subset', pheno_subset.shape)
print(pheno_subset.head())

# Take family id
family_id = pheno[cf.family_id]
X_train, y_train, X_test, y_test = divide_into_training_and_testing(pheno_subset, cpg, family_id=family_id)
print('X_train.shape, y_train.shape: ', X_train.shape, y_train.shape)
print('X_test.shape, y_test.shape: ', X_test.shape, y_test.shape)

# Calculate Z matrix
Z_train = create_familial_matrix(X_train, X_train['family_nb'])
print(Z_train)
print('Amount of 0.98: ', np.count_nonzero(Z_train == 0.98))

print('Shape of Z_train: ', Z_train.shape)

# Number of families
n_families = len(list(set(X_train['family_nb'])))
print('Amount of families', n_families)

# Drop family column and turn X into numpy array
print(X_train.drop(columns='family_nb').columns)
X_train = np.array(X_train.drop(columns='family_nb'))
y_train = np.array(y_train)


print('\n########################################')
print('Initializing hyperparameters')
tau_b = 2.0
Tau_b = 0.05
nu_b = 3.0


sigma_b, s_b, b = initialize_familial_parameters(n_fam=n_families, tau_b=tau_b, Tau_b=Tau_b, nu_b=nu_b)
plot_histogram(b, 'Ranfom effects')


# The inpendent individual error term
tau_e = 1.0
Tau_e = 0.01
nu_e = 2.0

sigma_e, s_e, scale_param = initialize_individual_parameters(n=X_train.shape[0], tau_e=tau_e, Tau_e=Tau_e, nu_e=nu_e)
plot_histogram(s_e, 'Mixture parameter of individual error term')
plot_histogram(scale_param, 'Scale parameter')


print('\n#####################################')
print('START CALCULATING THE BLUP')

p_dim = X_train.shape[1] + Z_train.shape[1]
coef_estimates = np.array([0.0001]*p_dim)

coefficient_matrix, right_hand = henderson_model_equations(y_train, X_train, Z_train, s_b, sigma_b, s_e, sigma_e)
print('Size of coefficient matrix: ', coefficient_matrix.shape)


#final_estimates = estimate_BLUP(coefficient_matrix=coefficient_matrix, coefficients=coef_estimates, right_hand=right_hand, n=200000)




