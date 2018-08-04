class Config(object):

    covariates = ['GENDER',
                'smoking',
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
                'PC5']

    # Random effect parameters for simulation
    simulation_params = {
        # Random effects parameters
        'tau_b' : 2.0,
        'Tau_b' : 0.05,
        'nu_b' : 3.0, # Degrees of freedom

        # Individual hyperparameters
        'tau_e' : 1.0,
        'Tau_e' : 0.01,
        'nu_e' : 2.0
    }





CONFIGS = Config()
