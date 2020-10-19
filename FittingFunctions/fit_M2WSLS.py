import numpy as np
from scipy.optimize import minimize

from LikelihoodFunctions.lik_M2WSLS import lik_M2WSLS


def fit_M2WSLS(actions, rewards):

    # to avoid division by 0 we don't want to include 0 and 1 in bounds
    bounds = [(1e-10, 1 - 1e-10)]

    # minimize risk of optimization failure
    fit_success = False
    max_iterations = 10
    while fit_success is False and max_iterations != 0:
        start_guess = [np.random.uniform(0, 1)]
        fitresult = minimize(
            lik_M2WSLS,
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