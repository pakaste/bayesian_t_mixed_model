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
from estimation.blup_estimates import estimate_BLUP
from estimation.variance_parameters import estimate_s_e
from estimation.variance_parameters import estimate_s_u
from estimation.variance_parameters import estimate_sigma_e
from estimation.variance_parameters import estimate_sigma_u
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
pheno_subset = subset_pheno(pheno, categorize=False, remove_correlated=True)
print('Shape of subset', pheno_subset.shape)
print(pheno_subset.head())

# Take family id
family_id = pheno[cf.family_id]
X_train, y_train, X_test, y_test = divide_into_training_and_testing(pheno_subset, cpg, family_id=family_id)
print('X_train.shape, y_train.shape: ', X_train.shape, y_train.shape)
print('X_test.shape, y_test.shape: ', X_test.shape, y_test.shape)

# Calculate Z matrix
family_ids = list(set(X_train['family_nb']))
Z_train, family_indices = create_familial_matrix(X_train, family_ids)
print('Shape of Z_train: ', Z_train.shape)

# Number of families
n_families = len(family_ids)
print('Amount of families', n_families)

# Drop family column and turn X into numpy array
X_train = X_train.drop(columns='family_nb')
X_train = np.array(X_train)
y_train = np.array(y_train)


print('\n########################################')
print('Initializing hyperparameters')
tau_b = 4.0
Tau_b = 3/8
nu_b = 2.0

sigma_b, s_b, b = initialize_familial_parameters(n_fam=n_families, tau_b=tau_b, Tau_b=Tau_b, nu_b=nu_b)
plot_histogram(b, 'Ranfom effects')


# The inpendent individual error term
tau_e = 4.0
Tau_e = 1/8
nu_e = 1.0

sigma_e, s_e, scale_param = initialize_individual_parameters(n=X_train.shape[0], tau_e=tau_e, Tau_e=Tau_e, nu_e=nu_e)
plot_histogram(s_e, 'Mixture parameter of individual error term')
plot_histogram(scale_param, 'Scale parameter')


print('\n#####################################')
print('START CALCULATING THE BLUP')

print('Average variance for b: ', np.mean(sigma_b))
print('Average variance for e: ', np.mean(sigma_e))

final_estimates, updated_s_e, updated_s_u, updated_sigma_e, updated_sigma_b = estimate_BLUP(y_train, X_train, Z_train, s_b, sigma_b, tau_b, Tau_b, nu_b, s_e, sigma_e, tau_e, Tau_e, nu_e, family_indices, n=10000)

for i in range(1, p_dim):
    estimation = round(np.mean(final_estimates[1000:, i]), 5)
    print('estimated params is {}'.format(estimation))

    plt.plot(final_estimates[2000:, i])
    plt.show()


