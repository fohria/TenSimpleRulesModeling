import numpy as np
from scipy.optimize import minimize

from LikelihoodFunctions.lik_M4CK import lik_M4CK


def fit_M4CK(actions, rewards):

    # to avoid division by 0 we don't want to include 0 and 1 in bounds
    # for exp(1), < 6e-5 of values are above 10, so 20 should be okay
    bounds = [(1e-10, 1 - 1e-10), (1e-10, 20)]

    fit_success = False
    max_iterations = 10
    while fit_success is False and max_iterations != 0:

        # alpha_c ~ uniform(0,1), beta_c ~ exponential(1)
        # we add + 0.5 to allow similar results as in the paper
        start_guess = [
            np.random.uniform(0, 1),  # alpha_c
            np.random.exponential(1) + 0.5  # beta_c
        ]
        fitresult = minimize(
            lik_M4CK,
            start_guess,
            args=(actions, rewards),
            bounds=bounds
        )
        fit_success = fitresult.success
        max_iterations -= 1

    parameter_estimations = fitresult.x
    loglikelihood = -fitresult.fun

    trial_count = len(actions)
    BIC = -2 * loglikelihood + len(start_guess) * np.log(trial_count)

    return parameter_estimations, loglikelihood, BIC
