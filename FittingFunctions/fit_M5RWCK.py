import numpy as np
from scipy.optimize import minimize

from LikelihoodFunctions.lik_M5RWCK import lik_M5RWCK


def fit_M5RWCK(actions, rewards):

    # to avoid division by 0 we don't want to include 0 and 1 in bounds
    # for exp(1), < 6e-5 of values are above 10, so 20 should be okay
    bounds = [
        (1e-10, 1 - 1e-10),  # alpha
        (1e-10, 20),         # beta
        (1e-10, 1 - 1e-10),  # alpha_c
        (1e-10, 20)          # beta_c
    ]

    fit_success = False
    max_iterations = 10
    while fit_success is False and max_iterations != 0:

        # alpha, alphac ~ uniform(0,1)
        # beta, betac ~ exponential(1)
        start_guess = [
            np.random.uniform(0, 1),        # alpha
            np.random.exponential(1),       # beta
            np.random.uniform(0, 1),        # alpha_c
            np.random.exponential(1) + 0.5  # beta_c (+ 0.5 as in paper's code)
        ]
        fitresult = minimize(
            lik_M5RWCK,
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
