import numpy as np
from scipy.optimize import minimize
from LikelihoodFunctions.lik_M4CK_v1 import lik_M4CK_v1


def fit_M4CK_v1(actions, rewards):

    # to avoid division by 0 we don't want to include 0 and 1 in bounds
    # for exp(1), < 6e-5 of values are above 10, so 20 should be okay
    bounds = [(1e-10, 1 - 1e-10), (1e-10, 20)]

    fit_success = False
    max_iterations = 10
    while fit_success is False and max_iterations != 0:
        # alphac ~ uniform(0,1), betac ~ exponential(1)
        # TODO: i guess ideally these guesses should be declared as optional
        # parameters to this function to allow more flexibility
        start_guess = [np.random.uniform(0, 1), np.random.exponential(1)+0.5]
        fitresult = minimize(
            lik_M4CK_v1, start_guess, args=(actions, rewards), bounds=bounds)
        fit_success = fitresult.success
        max_iterations -= 1

    parameter_estimations = fitresult.x
    loglikelihood = -fitresult.fun

    trial_count = len(actions)
    BIC = -2 * loglikelihood + len(start_guess) * np.log(trial_count)

    return parameter_estimations, loglikelihood, BIC
