from estimation.gibbs_sampler import run_one_chain


def run_multiple_chains(params, q):
    bayes_estimates = run_one_chain(y=params[0], X=params[1],
        Z=params[2], s_b=params[3], sigma_b=params[4],
        tau_b=params[5], Tau_b=params[6], nu_b=params[7],
        s_e=params[8], sigma_e=params[9], tau_e=params[10],
        Tau_e=params[11], nu_e=params[12],
        family_indices=params[13],
        n=params[14])

    #put the result in the Queue to return the the calling process
    q.put(bayes_estimates)