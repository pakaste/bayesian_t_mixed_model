import multiprocessing as mp
from numpy.random import normal
from estimation.gibbs_sampler import run_one_chain


def multiprocess(processes, params, iters, n_chains, normal_params=(0.0, 3.0)):
    """
        Uses multiprocessing for running multiple Gibbs sampler chains with different initial values.
        Parameters:
            processes       : (int) number of processes to be initialized
            params          : (list) [y_train, X_train, Z_train, s_b, sigma_b, tau_b, Tau_b, nu_b, s_e, sigma_e, tau_e, Tau_e, nu_e, family_indices]
            iters           : (int) number of iterations per chain
            n_chains        : (int) number of chains
            normal_params   : (tuple) mean and std of normal distribution
        Returns:
            (list of lists) of estimated parameters for Gibbs sampler for chains.
    """

    pool = mp.Pool(processes=processes)
    results = [pool.apply_async(run_one_chain, args=(params, iters, init_val)) for init_val in abs(normal(loc=normal_params[0], scale=normal_params[1], size=n_chains))]
    results = [p.get() for p in results]

    return results