from os.path import join
from os.path import abspath
from os.path import dirname
from os import pardir


class Config(object):

    CURRENT_DIR = abspath(dirname(__file__))
    ROOT_DIR = abspath(join(CURRENT_DIR, pardir))
    DATA_DIR = abspath(join(ROOT_DIR, "bayesian_t_mixed_model/data")) # Up to data/

    covariates = ['Age.at.blood.draw',
                #'GENDER',
                #'smoking',
                'CD8T',
                'CD4T',
                'NK',
                'Bcell',
                'Mono',
                'PC1_cp',
                'PC2_cp',
                'PC3_cp',
                'PC4_cp',
                'PC5_cp',
                'PC1',
                'PC2',
                'PC3',
                'PC4',
                'PC5',
                'family_nb']

    family_id = 'family_nb'

    # Random effect parameters for simulation
    SIMULATION_PARAMS = {
        # Random effects parameters
        'tau_b' : 4.0,
        'Tau_b' : 3/8,
        'nu_b' : 3.0, # Degrees of freedom

        # Individual hyperparameters
        'tau_e' : 4.0,
        'Tau_e' : 1/8,
        'nu_e' : 4.0
    }

    estimate_names = {
        0 : 'coefficients',
        1 : 's_e_estimates',
        2 : 's_u_estimates',
        3 : 'sigma_e_estimates',
        4 : 'sigma_b_estimates',
        5 : 'nu_e_estimates'
    }

    # Number of explanatory variables
    NVARS = 5

    # Number of individuals in training set
    TRAIN_SET = 730

    # GIBBS SAMPLER
    ITERS = 70000
    BURN_IN = 3000
    NCHAINS = 4 # Chains to be generated




CONFIGS = Config()
