import os
import numpy as np
import matplotlib.pyplot as plt

from data_wrangling.load_data import load_data, subset_pheno
from data_wrangling.simulate_data import generate_data, generate_parameters
from visualization.plotting import plot_histogram
from utils.hyperparameters import initialize_familial_parameters
from utils.hyperparameters import initialize_individual_parameters
from estimation.hendersson_equations import henderson_model_equations
from estimation.blup_estimates import estimate_BLUP


# Read in data
print('Working directory: ', os.getcwd())
cpg, pheno = load_data('data', all_cpgs=False)

print('Shape of cpg data: ', cpg.shape)
print('Shape of pheno data:', pheno.shape)
print(pheno.describe())


pheno_subset = subset_pheno(pheno, remove_nans=True)
print('Shape of subset without categorize: ', pheno_subset.shape)
print(pheno_subset)

pheno_subset = subset_pheno(pheno, remove_nans=True, categorize=True)
print('Shape of subset', pheno_subset.shape)









