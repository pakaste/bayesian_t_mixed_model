import os

# Set the OPENBLAS to use only one core
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import collections
import random
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
import multiprocessing as mp
from numpy.random import normal

from data_wrangling.load_data import load_data, subset_pheno
from data_wrangling.training_testing_set import divide_into_training_and_testing
from data_wrangling.training_testing_set import create_familial_matrix
from data_wrangling.simulate_data import simulate_fake_data
from visualization.plotting import plot_histogram

from utils.hyperparameters import initialize_familial_parameters
from utils.hyperparameters import initialize_individual_parameters

from estimation.gibbs_sampler import run_gibbs_sampler
from estimation.gibbs_sampler import run_one_chain

from estimation.variance_parameters import estimate_s_e
from estimation.variance_parameters import estimate_s_u
from estimation.variance_parameters import estimate_sigma_e
from estimation.variance_parameters import estimate_sigma_u
from settings import CONFIGS as cf

# Use real data
real = False

# Amount of CPUs to be used
n_cpu = 4

# Plot generated data
plot = False

# Gibbs sampler parameters
iters = 50000
burn_in = 2000

# Chains to be generated
n_chains = 4

# fake data
random.seed(10)
n_rows = 500
n_cols = 10

# Initialize hyperparameters
tau_b = 4.0   # degree of belief > 4
Tau_b = 3/8   # prior value for scale param
nu_b = 1/4.0

# The inpendent individual error term
tau_e = 4.0   # degree of belief > 2
Tau_e = 1/8   # prior value for scale
nu_e = 1/4.0  # [4, 10, 100, 1000]

# Get current datetime
dt_now = datetime.datetime.now()
year = str(dt_now.year)
month = str(dt_now.month)
day = str(dt_now.day)
dt_now_string = year + '_' + month + '_' + day

if real:
    # Read in data
    print('Working directory: ', os.getcwd())
    cpg, cpg_name, pheno = load_data('data', all_cpgs=False)
    cpg = pd.DataFrame(cpg)

    # take cpg name
    individual_id = cpg.index
    cpg.reset_index(inplace=True, drop=True)
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
    X_train.insert(0, 'Intercept', [1]*X_train.shape[0])
    X_train = X_train.iloc[:, :2]
    X_train = np.array(X_train)
    print('X_train.shape: ', X_train.shape)
    y_train = np.array(y_train)


    print('\n########################################')
    print('Initializing hyperparameters')
    sigma_b, s_b, b = initialize_familial_parameters(n_fam=n_families, tau_b=tau_b, Tau_b=Tau_b, nu_b=nu_b)
    sigma_e, s_e, scale_param = initialize_individual_parameters(family_indices, X_train.shape[0], tau_e=tau_e, Tau_e=Tau_e, nu_e=nu_e)

else:
    y_train, X_train, beta, Z_train, sigma_b, s_b, b, sigma_e, s_e, scale_param, family_indices = simulate_fake_data(n_rows, n_cols, tau_b, Tau_b, nu_b, tau_e, Tau_e, nu_e, seed=10)

if plot:
    plot_histogram(y_train, 'Dependent_variable')
    plot_histogram(b, 'Ranfom effects')
    plot_histogram(s_e, 'Mixture parameter of individual error term')
    plot_histogram(scale_param, 'Scale parameter')


print('\n#####################################')
print('START CALCULATING THE BLUP')

print('Average variance for b: ', np.mean(sigma_b))
print('Average variance for e: ', np.mean(sigma_e))
print('Nonzero elements in Z:', len(Z_train.nonzero()[0]))

# Put variables into a list
params = [y_train, X_train, Z_train, s_b, sigma_b, tau_b, Tau_b, nu_b, s_e, sigma_e, tau_e, Tau_e, nu_e, family_indices]

print('\n#####################################')
print('Start multiple chains')

pool = mp.Pool(processes=4)
results = [pool.apply_async(run_one_chain, args=(params, iters, init_val)) for init_val in abs(normal(loc=0.0, scale=3.0, size=n_chains))]
results = [p.get() for p in results]

# Create new folder for results
newpath = cf.DATA_DIR + '/' + dt_now_string

if not os.path.exists(newpath):
    os.makedirs(newpath)

for chain, result in enumerate(results):
    bayes_estimates = result
    final_estimates = bayes_estimates[0]
    s_e_estimates = bayes_estimates[1]
    s_u_estimates = bayes_estimates[2]
    sigma_e_estimates = bayes_estimates[3]
    sigma_b_estimates = bayes_estimates[4]
    nu_e_estimates = bayes_estimates[5]

    # Write results into a csv file
    for i in range(len(bayes_estimates)):
        pd_data = pd.DataFrame(bayes_estimates[i])
        file_name = newpath + '/' + cf.estimate_names[i] + '_chain_' + str(chain) + '.csv'
        pd_data.to_csv(file_name, header=False, index=False)
